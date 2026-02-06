import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class LogitWeightOptimizer(nn.Module):
    def __init__(self, num_models):
        super().__init__()
        # Initialize learnable weights for each model
        self.weights = nn.Parameter(torch.ones(num_models))
        
    def forward(self, stacked_logits):
        """
        Args:
            stacked_logits: Tensor of shape [batch_size, num_models, num_classes]
        Returns:
            weighted_logits: Tensor of shape [batch_size, num_classes]
        """
        # Apply softmax to weights to ensure they sum to 1
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # Reshape weights for broadcasting
        normalized_weights = normalized_weights.view(1, -1, 1)
        
        # Weight the logits from each model
        weighted_logits = (stacked_logits * normalized_weights).sum(dim=1)
        
        return weighted_logits

def train_weight_optimizer(dataset, model, optimizer, scheduler, loss_fn, num_epochs, args, print_every=20, logger=None):
    """
    Train the weight optimizer model.
    
    Args:
        dataset: Dataset that provides stacked logits and labels
        model: LogitWeightOptimizer instance
        optimizer: Optimizer for the weights
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        num_epochs: Number of training epochs
        args: Arguments containing setting and other configurations
        print_every: How often to print training progress
        logger: Logger instance for logging training progress
    """
    model.train()
    total_training_time = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = len(dataset.train_loader)
        
        for i, batch in enumerate(dataset.train_loader):
            start_time = time.time()
            
            # Get stacked logits and labels
            stacked_logits = batch['stacked_logits'].cuda()  # [batch_size, num_models, num_classes]
            labels = batch['labels'].cuda()
            
            # Forward pass
            weighted_logits = model(stacked_logits)
            
            # Calculate loss based on setting
            if args.setting == 'base2novel':
                # For base2novel, only compute loss on base classes
                loss = loss_fn(weighted_logits[:, dataset.base_class_idx], labels)
            else:
                # For few-shot, compute loss on all classes
                loss = loss_fn(weighted_logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            
            # Print progress
            if i % print_every == 0:
                avg_loss = total_loss / (i + 1)
                logger.info(f"Epoch {epoch} [{i}/{num_batches}] Loss: {avg_loss:.4f}")
                
                # Print current weights
                weights = F.softmax(model.weights, dim=0)
                weight_str = ", ".join([f"{w:.3f}" for w in weights])
                logger.info(f"Current weights: [{weight_str}]")
                
                # Print data time and batch time
                data_time = time.time() - start_time
                logger.info(f"Data time: {data_time:.3f}s")
        
        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        
        # Print epoch summary
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Print final weights for this epoch
        weights = F.softmax(model.weights, dim=0)
        weight_str = ", ".join([f"{w:.3f}" for w in weights])
        logger.info(f"Final weights for epoch {epoch}: [{weight_str}]")
        
        # Evaluate on validation set if available
        if hasattr(dataset, 'val_loader'):
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in dataset.val_loader:
                    stacked_logits = batch['stacked_logits'].cuda()
                    labels = batch['labels'].cuda()
                    
                    weighted_logits = model(stacked_logits)
                    
                    if args.setting == 'base2novel':
                        val_loss += loss_fn(weighted_logits[:, dataset.base_class_idx], labels).item()
                        pred = weighted_logits[:, dataset.base_class_idx].argmax(dim=1)
                    else:
                        val_loss += loss_fn(weighted_logits, labels).item()
                        pred = weighted_logits.argmax(dim=1)
                    
                    correct += (pred == labels).sum().item()
                    total += labels.size(0)
            
            val_loss /= len(dataset.val_loader)
            accuracy = 100. * correct / total
            logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
            model.train()
    
    logger.info(f"Training completed. Total training time: {total_training_time:.2f}s")
    return model
