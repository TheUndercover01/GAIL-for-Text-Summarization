from omegaconf import DictConfig

class Config():
        
    def generator_params(self):
        """
        Returns configuration parameters for the generator model.

        Returns:
            DictConfig: Configuration parameters for the generator.
        """
        self.genargs = {
            'generator_model_name_or_path': 'distilgpt2',
            # Assuming the reward model is part of the generator configuration
            'reward_model_name_or_path': 'distilbert-base-uncased',
            'seq_length': 512,  # Adjusted for smaller model capacity
            'batch_size': 16,  # Adjusted for GPU memory constraints
            'lr': 0.00006,  # Learning rate may need fine-tuning
            'prompt_size': 30,  # Keep as is or adjust based on task requirements
            'prompt_batch_size': 64,  # Adjusted for GPU memory constraints
            'num_rollouts': 64,  # Adjusted for computational efficiency
            'epochs': 100,  # Keep as per your experimentation needs
            'ppo_epochs': 4,  # Keep as per your experimentation needs
            'gen_kwargs': {
                'max_new_tokens': 40,  # Keep as is or adjust based on task requirements
                'top_k': 0,  # Sampling strategy parameter, keep as is
                'top_p': 1.0,  # Sampling strategy parameter, keep as is
                'do_sample': True  # Enable sampling in generation
            },
            'kl_coef': 0.01,  # Keep as is for KL-coefficient in loss calculation
            'gamma': 1,  # Discount factor for GAE
            'lam': 0.95,  # Lambda for GAE
            'cliprange': 0.2,  # PPO clipping range
            'cliprange_value': 0.2,  # Value clipping range
            'vf_coef': 1,  # Coefficient for value loss in PPO
        }

        # Return as a DictConfig object for OmegaConf compatibility
        genargs = DictConfig(self.genargs)
        return genargs
    
    def discriminator_params(self):
        """
        Returns configuration parameters for the discriminator model.

        Returns:
            DictConfig: Configuration parameters for the discriminator.
        """
        self.disargs = {
            "seed": 42,
            'model_name_or_path': 'distilbert-base-uncased',  # Updated to DistilBERT
            'learning_rate': 2e-5,  # Adjusted based on model and task
            'batch_size': 8,  # Adjusted for GPU memory constraints
            'gradient_accumulation_steps': 8,  # Adjusted for training stability
            'num_train_epochs': 3,  # Adjusted for sufficient training without overfitting
            'num_workers': 4,  # Adjust based on available CPU resources
            'seq_length': 512,  # Adjusted for smaller model capacity
            'logging_steps': 10,  # Log interval, keep as is or adjust based on preference
        }

        # Return as a DictConfig object for OmegaConf compatibility
        disargs = DictConfig(self.disargs)
        return disargs
