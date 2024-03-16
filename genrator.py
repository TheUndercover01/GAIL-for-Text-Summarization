import torch
import torch.nn as nn
from data_util import TextSummarizationDataGenerator
from transformers import #define the generator
import torch.nn.functional as F
import torch
import torch.nn as nn
from data_util import PPORLElement


class Agent(nn.Module):
    def __init__(self, model, args, trainable=False):
        """
        Initialize the Agent module.

        Args:
            model (PreTrainedModel): Pre-trained language model.
            args (dict): Arguments/configuration for the model.
            trainable (bool): Flag indicating whether the model is trainable.
        """
        super().__init__()
        self.trainable = trainable
        self.model = model  # Pre-trained language model
        
        # If not trainable, set the model to evaluation mode and freeze its parameters
        if not self.trainable:
            self.model = self.model.eval()
            self.model.requires_grad_(False)
        else:
            # Define value head for fine-tuning
            n_embd = self.model.lm_head.in_features
            num_labels = 1
            self.value_head = nn.Sequential(
                nn.LayerNorm(n_embd),
                nn.GELU(),
                nn.Linear(n_embd, 4*n_embd),
                nn.GELU(),
                nn.Linear(4*n_embd, num_labels),
            ).to(torch.bfloat16).to(self.model.device)
        
        # Logit head for generating outputs
        self.logit_head = self.model.get_output_embeddings()

    def generate(self, input_ids, **x):
        """
        Generate text summaries using the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            **x: Additional keyword arguments for generation.

        Returns:
            torch.Tensor: Generated text summaries.
        """
        return self.model.generate(input_ids, **x)  # Generate summary

    def forward(self, input_ids, attention_mask=None): 
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Logits (and value if trainable) from the model.
        """
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        lm_logits = self.logit_head(last_hidden_state)
        if self.trainable:
            # Calculate value for fine-tuning
            value = self.value_head(last_hidden_state).squeeze(-1)
            return lm_logits, value
        else:
            return lm_logits




class RolloutCreator():

    def __init__(
            self,
            article_dataset,
            args,
            tokenizer
    ):
        self.article_batch_size = args.article_batch_size
        self.article_dataset = article_dataset
        self.article_generator = TextSummarizationDataGenerator(self.article_dataset, self.article_batch_size)
        self.article_iterator = iter(self.article_generator)
        self.generate_kwargs = dict(
            args.gen_kwargs,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    def logprobs_from_logits(self ,logits, labels):
        logprobs = F.log_softmax(logits, dim=-1)
        logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
        return logprobs_labels.squeeze(-1)


    def make_experience(self, model, tokenizer, ref_model, reward_fn, args,num_rollouts=128):

        all_rollouts = []
        while len(all_rollouts) < num_rollouts:

            print(all_rollouts)
            try:
                batch = next(self.prompt_iterator)
            except StopIteration:
                self.article_generator = TextSummarizationDataGenerator(self.article_dataset, self.article_batch_size)
                self.article_iterator = iter(self.article_generator)
                batch = next(self.article_iterator)

            article_tensors = batch['input_ids'].to(model.model.device)


            trajectories = model.generate(
                article_tensors,
                attention_mask=batch['attention_mask'].to(model.model.device),
                **self.generate_kwargs
        )
            print(trajectories.shape ,  article_tensors.shape)
            summary_tensors = trajectories[:, article_tensors.shape[1]:]
            attention_mask = trajectories.not_equal(tokenizer.pad_token_id).long()

            with torch.no_grad():
                logits, values = model(
                    trajectories,
                    attention_mask=attention_mask,
                )
                ref_logits = ref_model(
                    trajectories,
                    attention_mask=attention_mask,
                )


            logprobs = self.logprobs_from_logits(logits[:, :-1, :], trajectories[:, 1:])
            ref_logprobs = self.logprobs_from_logits(ref_logits[:, :-1, :], trajectories[:, 1:])


            n_trajectories = trajectories.shape[0]
            values = values[:, :-1]




            start = batch['input_ids'].shape[1] - 1

            ends = start + attention_mask[:, start:].sum(1)

            truncated_values = [values[i, start : ends[i]] for i in range(n_trajectories)]
            truncated_logprobs = [logprobs[i, start : ends[i]] for i in range(n_trajectories)]

            texts = tokenizer.batch_decode(trajectories, skip_special_tokens=True)
            scores = reward_fn(texts)


            rewards = -args.kl_coef * (logprobs - ref_logprobs)

            all_rewards = [None] * n_trajectories
            for i in range(n_trajectories):
                rs = rewards[i][start : ends[i]]
                rs[-1] = scores[i]
                all_rewards[i] = rs

            new_rollout = [
                PPORLElement(
                    article_tensors=article_tensors[i],
                    summary_tensors=summary_tensors[i],
                    logprobs=truncated_logprobs[i],
                    values=truncated_values[i],
                    rewards=all_rewards[i],
                )
                for i in range(n_trajectories)
            ]
            all_rollouts += new_rollout

        score = torch.tensor(scores).mean().detach().cpu().item()

        return all_rollouts, score