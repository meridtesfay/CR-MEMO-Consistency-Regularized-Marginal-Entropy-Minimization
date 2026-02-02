import torch
import torch.optim as optim
from src.losses import ConsistencyRegularizedLoss # Assumes the loss extracted previously

class MEMOAdapter:
    def __init__(self, model, lr=0.0002, steps=1, kl_weight=0.1):
        """
        MEMO: Marginal Entropy Minimization with One test point.
        """
        self.model = model
        self.lr = lr
        self.steps = steps
        self.kl_weight = kl_weight

    def adapt_and_predict(self, x_augments):
        """
        Adapts model parameters on a single sample (multiple augmented views)
        and returns the final prediction.
        
        Args:
            x_augments: A batch of B augmentations of a single image.
        """
        # We only adapt specific parameters (like BatchNorm) to prevent overfitting
        params_to_update = []
        for name, param in self.model.named_parameters():
            if "bn" in name: 
                params_to_update.append(param)
        
        optimizer = optim.SGD(params_to_update, lr=self.lr)
        criterion = ConsistencyRegularizedLoss(kl_weight=self.kl_weight)

        for _ in range(self.steps):
            optimizer.zero_grad()
            outputs = self.model(x_augments)
            
            # Using your custom Consistency-Regularized Loss
            loss = criterion(outputs)
            
            loss.backward()
            optimizer.step()

        # Inference: Return the average prediction across augmentations
        with torch.no_grad():
            final_outputs = self.model(x_augments)
            avg_probs = torch.softmax(final_outputs, dim=1).mean(dim=0)
            
        return avg_probs