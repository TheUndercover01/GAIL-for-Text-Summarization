import torch
from genrator import RolloutCreator


class LossCalculator():
    """
    Class for calculating the loss function and advantages for PPO training.

    Args:
        model: The PPO model used for training.
        tokenizer: Tokenizer object for tokenizing the data.
        args: Arguments for configuring the loss calculator.

    Attributes:
        model: The PPO model used for training.
        tokenizer: Tokenizer object for tokenizing the data.
        args: Arguments for configuring the loss calculator.
    """

    def __init__(self, model, tokenizer, args):
        """
        Initializes the LossCalculator.

        Args:
            model: The PPO model used for training.
            tokenizer: Tokenizer object for tokenizing the data.
            args: Arguments for configuring the loss calculator.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def calculate_loss(self, mini_batch):
        """
        Calculates the PPO loss function.

        Args:
            mini_batch: Mini-batch containing the query tensors, response tensors, log probabilities, values, and rewards.

        Returns:
            Tuple[torch.Tensor, float]: Tuple containing the loss tensor and the average reward.
        """
        article_tensor = mini_batch.article_tensor
        summary_tensor = mini_batch.summary_tensor
        old_logprobs = mini_batch.logprobs
        old_values = mini_batch.values
        old_rewards = mini_batch.rewards

        response_length = old_rewards.shape[1]

        advantages, returns = self.calculate_gae(old_values, old_rewards)

        trajectories = torch.hstack([mini_batch.article_tensor, mini_batch.summary_tensor])
        attention_mask = trajectories.not_equal(self.tokenizer.pad_token_id).long()
        logits, values_pred = self.model(trajectories, attention_mask=attention_mask)

        values_pred = values_pred[:, :-1]
        logprobs = self.RolloutCreator.logprobs_from_logits(logits[:, :-1, :], trajectories[:, 1:])
        attention_mask = attention_mask[:, :-1]

        start = article_tensor.shape[1] - 1
        end = start + response_length
        logprobs, values_pred, mask = (
            logprobs[:, start:end],
            values_pred[:, start:end],
            attention_mask[:, start:end],
        )

        loss = self.ppo_loss(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
        )

        return loss, old_rewards[:,-1].mean().item()

    def ppo_loss(self, logprobs, values, old_logprobs, old_values, advantages, returns, mask):
        """
        Computes the PPO loss.

        Args:
            logprobs: Log probabilities of the actions.
            values: Predicted values.
            old_logprobs: Old log probabilities of the actions.
            old_values: Old predicted values.
            advantages: Advantage values.
            returns: Estimated returns.
            mask: Mask indicating valid positions in the sequence.

        Returns:
            torch.Tensor: PPO loss tensor.
        """
        values_clipped = torch.clamp(
            values,
            old_values - self.args.cliprange_value,
            old_values + self.args.cliprange_value,
        )
        n = mask.sum()
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n
        loss = pg_loss + self.args.vf_coef * vf_loss
        return loss

    def calculate_gae(self, values, rewards):
        """
        Calculates the Generalized Advantage Estimation (GAE) values.

        Args:
            values: Predicted values.
            rewards: Rewards received.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the advantage values and estimated returns.
        """
        advantages = torch.zeros_like(rewards, device=rewards.device)
        last_advantage = 0
        last_value = 0
        with torch.no_grad():
            for t in reversed(range(rewards.shape[1])):
                delta = rewards[:, t] + self.args.gamma * last_value - values[:, t]
                last_advantage = delta + self.args.gamma * self.args.lam * last_advantage
                advantages[:, t] = last_advantage
                last_value = values[:, t]
            returns = advantages + values
        return advantages, returns

  