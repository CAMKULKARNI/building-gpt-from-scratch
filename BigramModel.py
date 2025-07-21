import tensorflow as tf
import keras
from tqdm import tqdm

# --- Hyperparameters & Constants ---
INPUT_FILE = "input.txt"
TRAIN_SPLIT_RATIO = 0.9
BLOCK_SIZE = 8  # Maximum context length for predictions
BATCH_SIZE = 32 # Number of independent sequences to process in parallel
NUM_STEPS = 30000 # Number of training iterations
EVAL_INTERVAL = 100 # How often to evaluate on the validation set
LEARNING_RATE = 1e-2 # Learning rate for the Adam optimizer


# --- Data Loading and Preprocessing ---

# Load the dataset from the file
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    text = file.read()

# Create the vocabulary from the unique characters in the text
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)

# Create mappings from characters to integers and vice-versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Define encoder and decoder functions
encode = lambda s: [stoi[ch] for ch in s] # encoder: take a string, output a list of integers
decode = lambda l: "".join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Convert the entire dataset to a TensorFlow tensor
data = tf.convert_to_tensor(encode(text), dtype=tf.int64)

# Split the dataset into training and validation sets
n = int(TRAIN_SPLIT_RATIO * len(data))
train_data = data[:n]
val_data = data[n:]


# --- Data Batching ---

def get_batch(split):
    """
    Generate a small batch of data of inputs x and targets y.
    
    Args:
        split (str): 'train' or 'val' to select the dataset.
        
    Returns:
        A tuple of (x, y) tensors.
    """
    current_data = train_data if split == "train" else val_data
    # Generate random starting indices for the batches
    ix = tf.random.uniform((BATCH_SIZE,), maxval=current_data.shape[0] - BLOCK_SIZE, dtype=tf.int64)
    
    # Create input sequences (x) and target sequences (y)
    x_batch = tf.stack([current_data[i:i + BLOCK_SIZE] for i in ix])
    y_batch = tf.stack([current_data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x_batch, y_batch


# --- Model Definition ---

class BigramLanguageModel(keras.Model):
    """
    A simple Bigram Language Model using a Keras Embedding layer.
    """
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table.
        self.embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=vocab_size)
    
    def call(self, idx, targets=None):
        """
        Forward pass for the model.
        
        Args:
            idx (tf.Tensor): Input tensor of shape [batch_size, block_size].
            targets (tf.Tensor, optional): Target tensor of shape [batch_size, block_size]. 
                                           If provided, computes the loss. Defaults to None.
                                           
        Returns:
            A tuple of (logits, loss). Loss is None if targets are not provided.
        """
        # idx and targets are both tensors of shape [batch_size, block_size]
        logits = self.embedding(idx) # shape [batch_size, block_size, vocab_size]
        
        loss = None
        if targets is not None:
            # Keras's sparse_categorical_crossentropy expects logits of shape
            # [batch_size, sequence_length, num_classes] and targets of shape
            # [batch_size, sequence_length], which matches our data.
            loss = keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens based on a starting context.
        
        Args:
            idx (tf.Tensor): A tensor of shape [batch_size, sequence_length] representing the context.
            max_new_tokens (int): The maximum number of new tokens to generate.
            
        Returns:
            A tensor of shape [batch_size, sequence_length + max_new_tokens] with the generated sequence.
        """
        # idx is a tensor of shape [batch_size, sequence_length]
        for _ in range(max_new_tokens):
            # For a bigram model, we only need the last token to predict the next one.
            # However, we crop to block_size to keep the input shape consistent for potential
            # future upgrades to more complex models (like Transformers).
            idx_cond = idx[:, -BLOCK_SIZE:]
            
            # Get the predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step's logits
            logits = logits[:, -1, :] # becomes [batch_size, vocab_size]
            
            # Sample from the distribution defined by the logits.
            # tf.random.categorical is more efficient when given logits directly.
            idx_next = tf.random.categorical(logits, num_samples=1) # shape [batch_size, 1]

            # Append the sampled index to the running sequence
            idx = tf.concat([idx, idx_next], axis=-1) # shape [batch_size, sequence_length + 1]
            
        return idx


# --- Main Execution ---

def main():
    """
    Main function to run the data processing, model training, and text generation.
    """
    print("--- Data Information ---")
    print(f"Length of the dataset in characters: {len(text)}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(" ".join(chars))
    print("-" * 25)

    # Instantiate the model
    model = BigramLanguageModel(VOCAB_SIZE)

    # Check initial performance before training
    xb, yb = get_batch("train")
    _, loss_initial = model(xb, yb)
    print(f"Initial loss (before training): {loss_initial.numpy():.4f}")

    # Generate text from the untrained model for comparison
    print("\n--- Generating text from UNTRAINED model ---")
    context_untrained = tf.zeros((1, 1), dtype=tf.int64)
    output_untrained = model.generate(context_untrained, max_new_tokens=100)[0].numpy().tolist()
    print(decode(output_untrained))
    print("-" * 25)

    # --- Train the model ---
    print("\n--- Starting model training ---")
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Training loop
    bar = tqdm(range(NUM_STEPS))
    val_loss = tf.constant(0.0) # Initialize val_loss
    for step in bar:
        # Get a batch of training data
        x_batch, y_batch = get_batch("train")

        # Evaluate the loss and compute gradients
        with tf.GradientTape() as tape:
            _, loss = model(x_batch, y_batch)
        
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Periodically evaluate validation loss
        if step % EVAL_INTERVAL == 0 or step == NUM_STEPS - 1:
            x_val_batch, y_val_batch = get_batch("val")
            _, val_loss = model(x_val_batch, y_val_batch)
        
        bar.set_postfix(train_loss=loss.numpy(), val_loss=val_loss.numpy())

    print(f"\nTraining finished. Final training loss: {loss.numpy():.4f}")
    print(f"Final validation loss: {val_loss.numpy():.4f}")
    print("-" * 25)

    # --- Generate text from the TRAINED model ---
    print("\n--- Generating text from TRAINED model ---")
    context_trained = tf.zeros((1, 1), dtype=tf.int64) 
    generated_output_trained = model.generate(idx=context_trained, max_new_tokens=500)
    decoded_text_trained = decode(generated_output_trained[0].numpy().tolist())
    print(decoded_text_trained)
    print("\nScript finished.")

if __name__ == "__main__":
    main()