from transformers import GenerationConfig
import datetime
from datetime import timezone
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import os
from typing import Callable, Optional, Dict
import shutil
import json
from transformers.trainer_utils import is_main_process
import wandb
import torch
from state_manager import get_state, set_state
MAX_TRIES = 9


MIS_MATCH_VOCAB_SIZE_MODELS = [
    'NousResearch/Nous-Capybara-7B-V1',
    'berkeley-nest/Starling-LM-7B-alpha',
    'NousResearch/Hermes-2-Theta-Llama-3-8B',
    'MNC-Jihun/Mistral-7B-AO-u0.5-b2-ver0.4'
]

ERROR_GENERATION_CONFIG_MODELS = [
    "lmsys/vicuna-7b-v1.5", 
    "lmsys/vicuna-13b-v1.5",
    "NousResearch/Nous-Hermes-llama-2-7b", 
    "defog/llama-3-sqlcoder-8b"
]

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

print(f"posto={LOCAL_RANK}", flush=True)
    
class CustomEvalSaveCallback(TrainerCallback):
    def __init__(
        self,
        function_when_to_evaluate: Callable,
        submission_dir: str,
        output_dir: str,
        original_model_name: str,
        max_steps: int = -1,
        checking_step: int = 100,
        total_steps_all_epochs: int = -1,
        end_time: str = "",
        checking_mode: str = "none",
        use_reward_accuracy: bool = False,
    ):
        self.function_when_to_evaluate = function_when_to_evaluate
        self.submission_dir = submission_dir
        self.current_best_loss = None
        self.best_checkpoint_info = None
        self.update_best_checkpoint = False
        self.output_dir = output_dir
        self.original_model_name = original_model_name
        self.max_steps = max_steps
        self.has_checkpoint = False
        self.save_only = False
        self.checking_step = checking_step
        self.total_steps_all_epochs = total_steps_all_epochs
        self.checking_mode = checking_mode
        # Cache the original model's config for architecture patching
        self._original_config = None
        base_config_path = os.path.join(original_model_name, "config.json")
        if os.path.exists(base_config_path):
            try:
                with open(base_config_path) as f:
                    self._original_config = json.load(f)
            except Exception:
                pass
        self.end_time = end_time
        self.use_reward_accuracy = use_reward_accuracy
        self._eval_timing_adjusted = False

    def _patch_submission_architectures(self):
        """Restore original architectures in submission config.json after copy."""
        if not self._original_config:
            return
        config_path = os.path.join(self.submission_dir, "config.json")
        if not os.path.exists(config_path):
            return
        orig_arch = self._original_config.get("architectures")
        if not orig_arch:
            return
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            if cfg.get("architectures") != orig_arch:
                cfg["architectures"] = orig_arch
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
        except Exception:
            pass

    def compute_loss(self, state: TrainerState, metrics):
        if self.use_reward_accuracy:
            acc = metrics.get("eval_rewards/accuracies")
            if acc is not None:
                return -acc  # negate: callback minimizes, but higher accuracy = better
        return metrics.get("eval_loss", None)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Custom logic to decide whether to save or evaluate
        # print(f"************* on_step_end: {state.global_step}, check eval", flush=True)
        # TODO: implement the logic to save the model without evaluating if there is no check points --> avoid evaluating takes too much time
        # Check if the checking_step is reached
        # print(f"Checking the model at step: {state.global_step}, checking_step: {self.checking_step}, checking_mode: {self.checking_mode}", flush=True)
        if state.global_step == self.checking_step and self.checking_mode == "first_time":
            # print(f"Checking the model at step: {state.global_step}", flush=True)
            # check the time so far to estimate the training time in total 
            my_state = get_state()
            start_time_obj = datetime.datetime.strptime(my_state["train"]["start_time"], "%Y-%m-%d %H:%M:%S")
            start_train_time_obj = datetime.datetime.strptime(my_state["train"]["start_train_time"], "%Y-%m-%d %H:%M:%S")
            
            log_content = f"verificando passo={state.global_step}"
            now = datetime.datetime.now()
            preparation_time = (start_train_time_obj - start_time_obj).total_seconds()
            log_content += f"\npreparação={preparation_time}s"
            time_so_far = (now - start_time_obj).total_seconds()
            log_content += f"\ndecorrido={time_so_far}s"
            time_for_one_step = (now - start_train_time_obj).total_seconds() / self.checking_step
            log_content += f"\ntempo/passo={time_for_one_step}s"
            # Now estimate the total training time for this training
            log_content += f"\npassos_total={self.total_steps_all_epochs}"
            total_remaining_training_time = time_for_one_step * (self.total_steps_all_epochs - state.global_step)
            log_content += f"\ntreino_restante={total_remaining_training_time}s"
            # n * time_so_far + total_remaining_training_time = total_remaining_time
            end_time_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
            total_remaining_time = (end_time_obj - now).total_seconds()
            log_content += f"\ntempo_restante={total_remaining_time}s"
            
            # n * time_so_far + (time_so_far + total_remaining_training_time) = total_remaining_time
            # time_so_far + total_remaining_training_time is the time it takes to finish the training (need to estimate the eval time and save time, assuming this is 15 minutes)
            # assuming time_so_far is + 5 minutes, just in case the checking step takes more time than expected
            max_var_time_sofar = 3 * 60
            n = (total_remaining_time - (time_so_far + total_remaining_training_time + 12 * 60)) / (time_so_far + max_var_time_sofar) # 300 = 5 minutes, assume that it extra time would be more or less 5 minutes
            n = int(n)
            my_state["check_details"] = {
                "now": str(now.strftime("%Y-%m-%d %H:%M:%S")),
                "start_time": str(start_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "start_train_time": str(start_train_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "checking_step": self.checking_step,
                "checking_mode": self.checking_mode,
                "estimation_of_steps": n,
                "preparation_time": preparation_time,
                "time_so_far": time_so_far,
                "time_for_one_step": time_for_one_step,
                "total_remaining_training_time": total_remaining_training_time,
                "total_remaining_time": total_remaining_time,
                "end_time": self.end_time,
            }
            if n > 0: # we should try more 
                log_content += f"\npassos_estimados={n}"
                control.should_training_stop = True
                control.should_save = False
                args.save_strategy = "no"
                # save the current loss of this step to the state;
                last_log = state.log_history[-1]
                my_state["train"]["current_loss"] = last_log["loss"]
                my_state["mode"] = "continue"
                if n > MAX_TRIES:
                    n = MAX_TRIES
                log_content += f"\ntentativas={n + 1}"
                my_state["next_runs"] = n + 1 # including the current run
            else:
                print(f"sem tempo, terminando", flush=True)
                my_state["mode"] = "finish"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                print(log_content, flush=True)            
            return control
    
        elif state.global_step == self.checking_step and self.checking_mode == "second_time": # at second time, we don't estimate the training time again, just save the current_loss
            log_content = f"segunda verificação, passo={state.global_step}"            
            my_state = get_state()
            current_loss = state.log_history[-1]["loss"]
            my_state["train"]["current_loss"] = current_loss
                
            control.should_training_stop = True

            # Check if current_loss > current min_loss --> do not save to save time and space
            # 
            # if my_state["train"]["current_loss"] > current_min_loss:
            #     print(f"Current loss: {my_state['train']['current_loss']} is greater than the current min_loss: {current_min_loss}, do not save the checkpoint", flush=True)
            #     control.should_save = False
            # check if this is the last run and the current_loss is the lowest --> keep running the training
            current_is_the_best = False
            current_min_loss = min([run["current_loss"] for run in my_state["runs"]])
            if current_loss <= current_min_loss:
                if len(my_state["runs"]) + 1 == my_state["next_runs"]:
                    print(f"perda={my_state['train']['current_loss']} < mínimo={current_min_loss}, melhor até agora", flush=True)
                    current_is_the_best = True
                    
            if current_is_the_best:
                control.should_training_stop = False
                my_state["mode"] = "finish"
            else:
                control.should_save = False
                args.save_strategy = "no"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                # print(log_content, flush=True)
        
            
        when_to_eval = self.function_when_to_evaluate(state.global_step)
        if when_to_eval["eval"]:
            # do not allow the pod to be stopped by any reason
                # first check if there is at least one checkpoint or not
            print(f"avaliando passo={state.global_step}, motivo={when_to_eval['reason']}", flush=True)
            control.should_evaluate = True
            control.should_save = True
            if when_to_eval["reason"] == "end_time":
                # Final wall-clock save: STOP training afterwards so
                # trainer.train() returns (the dev-pass and success.txt live
                # after it — epoch planning deliberately over-plans by 1.25x, so
                # without this stop the loop runs until the external kill and
                # they never execute). HF processes should_evaluate/should_save
                # for this step BEFORE honouring should_training_stop, so the
                # eval+save still happen.
                control.should_training_stop = True
                if not self.has_checkpoint: # if there is no checkpoint, we just save the model, do not evaluate
                    print(f"sem checkpoint, só salvando no passo={state.global_step}", flush=True)
                    control.should_evaluate = False
                    self.save_only = True

        # Skip evals before 0.75 epochs — no overfitting possible yet,
        # save compute for frequent post-epoch-1 evals instead.
        # Only skip when training for >1 epoch; sub-epoch runs need every eval.
        # NEVER skip the end_time save: its trigger is one-shot (run_eval flag),
        # so cancelling it here would silently lose the final save entirely.
        if control.should_evaluate and self.total_steps_all_epochs > 0:
            steps_per_epoch = self.total_steps_all_epochs / max(1, args.num_train_epochs)
            if (
                when_to_eval["reason"] != "end_time"
                and args.num_train_epochs > 1
                and state.global_step < int(0.75 * steps_per_epoch)
            ):
                control.should_evaluate = False
                control.should_save = False

        return control


    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs
    ):
        self.save_only = False
        # Append eval_loss to file
        eval_loss = self.compute_loss(state, metrics)
        if state.global_step < 2:
            return 
        print(f"examinando passo={state.global_step}", flush=True)
        if eval_loss is None:
            print(f"perda nula no passo={state.global_step}, ignorando", flush=True)
            return
        if self.best_checkpoint_info is None or eval_loss < self.best_checkpoint_info["loss"]:
            print(f"novo melhor no passo={state.global_step}, perda={eval_loss}", flush=True)
            self.best_checkpoint_info = {
                "loss": eval_loss,
                "step": state.global_step
            }
            self.update_best_checkpoint = True
        else:
            if self.best_checkpoint_info is not None:
                print(f"passo={state.global_step}: perda={eval_loss} >= melhor={self.best_checkpoint_info['loss']}, sem atualização", flush=True)

        # After first real eval: use measured runtime to adjust eval frequency.
        # Budget 10% of remaining time for evals. Only widens the interval (never shrinks).
        if not self._eval_timing_adjusted and self.end_time and self.total_steps_all_epochs > 0:
            self._eval_timing_adjusted = True
            eval_runtime = metrics.get("eval_runtime", 0)
            if eval_runtime > 0:
                try:
                    end_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
                    remaining_s = max(0, (end_obj - datetime.datetime.now()).total_seconds())
                except (ValueError, TypeError):
                    remaining_s = 0
                if remaining_s > 0:
                    eval_budget_s = remaining_s * 0.10
                    max_evals = max(3, int(eval_budget_s / eval_runtime))
                    new_eval_steps = max(30, self.total_steps_all_epochs // max_evals)
                    if new_eval_steps > args.eval_steps:
                        print(
                            f"Espelho demorou {eval_runtime:.1f}s, ajustando vaidade pra cada {new_eval_steps} passos",
                            flush=True,
                        )
                        args.eval_steps = new_eval_steps
                        args.save_steps = new_eval_steps
                    else:
                        print(f"Espelho rápido ({eval_runtime:.1f}s), mantendo o ritmo", flush=True)

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        
        if state.global_step == self.max_steps and self.max_steps != -1:
            print(f"teto atingido: {self.max_steps} passos", flush=True)
            control.should_training_stop = True

        self.has_checkpoint = True
        
        if not is_main_process(LOCAL_RANK): # if not main process, skip this
            return 
            
        if self.save_only: # if only save, do not evaluate 
            print(f"só salvando, sem avaliação, passo={state.global_step}", flush=True)
            current_step = state.global_step
            # Remove existing directory if it exists
            if os.path.exists(self.submission_dir):
                shutil.rmtree(self.submission_dir)
                
            shutil.copytree(
                os.path.join(self.output_dir, f"checkpoint-{current_step}"),
                self.submission_dir
            )
            self._patch_submission_architectures()
            self.update_best_checkpoint = False
            # add a loss.txt file to the submission directory
            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{current_step},no_eval")

            # release the flag
            self.save_only = False
            return 
            
        # Custom logic after model is saved
        # You can trigger external services, logs, or backups here
        if (
            self.update_best_checkpoint
            and is_main_process(LOCAL_RANK)
        ):
            print(f"enviando melhor checkpoint, passo={state.global_step}", flush=True)
            # Remove existing directory if it exists
            if os.path.exists(self.submission_dir):
                shutil.rmtree(self.submission_dir)
            best_eval_loss = self.best_checkpoint_info["loss"]
            shutil.copytree(
                os.path.join(self.output_dir, f"checkpoint-{self.best_checkpoint_info['step']}"),
                self.submission_dir
            )
            self._patch_submission_architectures()
            self.update_best_checkpoint = False
            # add a loss.txt file to the submission directory
            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{self.best_checkpoint_info['step']},{best_eval_loss}")


VALIDATOR_BETA_GRPO = 0.5


class GRPOCustomEvalSaveCallback(CustomEvalSaveCallback):
    def compute_loss(self, state: TrainerState, metrics):
        eval_reward, eval_kl = self._extract_metrics(state, metrics)
        if eval_reward is None:
            print("recompensa não encontrada nas métricas", flush=True)
            return None

        if eval_kl is not None:
            score = eval_reward - VALIDATOR_BETA_GRPO * eval_kl
            print(
                f"pontuação: recompensa({eval_reward:.4f}) - {VALIDATOR_BETA_GRPO}*kl({eval_kl:.4f}) = {score:.4f}",
                flush=True,
            )
        else:
            score = eval_reward
            print(
                f"pontuação sem kl: recompensa({eval_reward:.4f})",
                flush=True,
            )

        return -score

    @staticmethod
    def _extract_metrics(state: TrainerState, metrics):
        sources = []
        if metrics:
            sources.append(metrics)
        if state.log_history:
            sources.append(state.log_history[-1])

        eval_reward, eval_kl = None, None
        kl_keys = ("eval_kl", "eval_completions/mean_kl", "eval/kl", "eval_mean_kl")
        for src in sources:
            if eval_reward is None:
                eval_reward = src.get("eval_reward")
            if eval_kl is None:
                for k in kl_keys:
                    if k in src:
                        eval_kl = src[k]
                        break
        return eval_reward, eval_kl
    
    def penalize_eval_loss(self, eval_loss: float):
        if eval_loss < 0:
            return eval_loss / 3
        else:
            return eval_loss * 3


def check_remaining_time_less_than_minutes(end_time: str, minutes: int) -> bool: 
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)  # Make end_time timezone-aware in UTC
    now = datetime.datetime.now(timezone.utc)
    time_diff = end_time - now
    result =  time_diff.total_seconds() < minutes * 60
    if result:
        print(f"agora={now} fim={end_time} diff={time_diff}", flush=True)
    return result


class WhenToEvalHandler:
    def __init__(self, end_time: str, save_before_remaining_time: int = 3, periodic_save_steps: int = -1, steps_per_epoch: int = -1, max_steps: int = -1):
        self.save_before_remaining_time = save_before_remaining_time
        self.run_eval = False
        self.end_time = end_time
        self.periodic_save_steps = periodic_save_steps
        self.steps_per_epoch = steps_per_epoch
        self.max_steps = max_steps

    def __call__(self, global_step: int) -> dict:
        
        if self.steps_per_epoch != -1 and global_step % self.steps_per_epoch == 0 and global_step > 1:
            return {"eval": True, "reason": "epoch"}
        
        if self.periodic_save_steps != -1 and global_step % self.periodic_save_steps == 0 and global_step > 1:
            return {"eval": True, "reason": "periodic"}
        
        if self.save_before_remaining_time > 0 and not self.run_eval:
            if check_remaining_time_less_than_minutes(self.end_time, self.save_before_remaining_time):
                print(f"tempo acabando, avaliando e salvando", flush=True)
                # the eval time might be higher than the end_time, so we need to let the pod not stop by setting a flag for this
                self.run_eval = True
                return {"eval": True, "reason": "end_time"}
        
        if self.max_steps != -1 and global_step == self.max_steps:
            print(f"teto atingido: {self.max_steps} passos", flush=True)
            return {"eval": True, "reason": "max_step"}

        return {"eval": False, "reason": "none"}


def set_generation_config(model_name, model):
    try:
        if model_name in ERROR_GENERATION_CONFIG_MODELS:
            model.generation_config = GenerationConfig(temperature=None, top_p=None)
    except:
        print(f"falha ao configurar geração para {model_name}")
        pass


def resize_if_needed(model_name, model, token_nums):
    try:
        if model_name in MIS_MATCH_VOCAB_SIZE_MODELS:
            model.resize_token_embeddings(token_nums)
    except:
        print(f"falha ao redimensionar tokens para {model_name}")
        pass


def init_wandb(train_request: Dict):
    # set wandb_mode=offline; do not upload the data to wandb export WANDB_MODE=offline
    return True
    task_id = train_request["task_id"]
    expected_repo_name = train_request["expected_repo_name"]
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = train_request["wandb_log_dir"]
    os.environ["WANDB_RUN_ID"] = f"{task_id}_{expected_repo_name}"
    os.environ["WANDB_NAME"] = f"{task_id}_{expected_repo_name}"
    if is_main_process(LOCAL_RANK):
        os.makedirs(train_request["wandb_log_dir"], exist_ok=True)
    return True