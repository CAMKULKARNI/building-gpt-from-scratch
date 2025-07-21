import tensorflow as tf
import keras
from tqdm import tqdm

# --- Hyperparameters & Constants ---
INPUT_FILE = "input.txt"
TRAIN_SPLIT_RATIO = 0.9
BLOCK_SIZE = 256  # Maximum context length for predictions
BATCH_SIZE = 8 # Number of independent sequences to process in parallel
NUM_ITERATIONS = 10000 # Number of training iterations
EVAL_INTERVAL = 100 # How often to evaluate on the validation set
LEARNING_RATE = 3e-4 # Learning rate for the Adam optimizer
N_EMBED = 256 # Dimensionality of the embedding space


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

class Head(keras.layers.Layer):
    """
    A single head of self-attention.
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = keras.layers.Dense(head_size, use_bias=False)  # key projection
        self.query = keras.layers.Dense(head_size, use_bias=False)  # query projection
        self.value = keras.layers.Dense(head_size, use_bias=False)  # value projection
        # self.tril is not a parameter of the model, so we make it a non-trainable buffer.
        # In PyTorch, this would be self.register_buffer('tril', ...)
        tril_const = tf.linalg.band_part(tf.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=tf.float32), -1, 0)
        self.tril = tf.Variable(tril_const, trainable=False)

    def call(self, x: tf.Tensor):
        B, T, C = x.shape  # B is batch size, T is sequence length, C is embedding dimension
        k = self.key(x)  # shape [B, T, head_size]
        q = self.query(x)  # shape [B, T, head_size]
        v = self.value(x)  # shape [B, T, head_size]
        weights = (k @ tf.transpose(q, perm=[0, 2, 1])) * (C ** (-0.5))  # shape [B, T, T]
        weights = tf.where(self.tril[:T, :T] == 0, tf.convert_to_tensor(float("-inf")), weights)
        weights = tf.nn.softmax(weights, axis=-1)
        output = weights @ v  # shape [B, T, head_size]
        return output


class MultiHeadAttention(keras.layers.Layer):
    """
    Multiple heads of self-attention in parallel.
    """
    def __init__(self, n_heads, head_size, training=True):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(n_heads)]
        self.proj = keras.layers.Dense(N_EMBED)  # Final projection to combine heads
        self.dropout = keras.layers.Dropout(rate=0.1)  # Dropout layer for regularization
        self.training = training  # Whether the model is in training mode

    def call(self, x: tf.Tensor):
        # Apply each head and concatenate the results
        output = tf.concat([head(x) for head in self.heads], axis=-1)  # shape [B, T, n_heads * head_size]
        output = self.proj(output)  # shape [B, T, n_heads * head_size]
        return self.dropout(output, training=self.training)  # Apply dropout if in training mode

# Per token level
class FeedForward(keras.layers.Layer):
    """
    A simple feed-forward layer with a ReLU activation.
    """
    def __init__(self, n_embed, training=True):
        super().__init__()
        # The original paper uses a 4x expansion in the feed-forward layer. Hence n_embed * 4.
        self.linear = keras.layers.Dense(n_embed * 4)
        self.swish = keras.layers.Activation('swish')
        self.linear_out = keras.layers.Dense(n_embed)  # Output layer to match input dimension
        self.dropout = keras.layers.Dropout(rate=0.1)  # Dropout layer for regularization
        self.training = training  # Whether the model is in training mode

    def call(self, x: tf.Tensor):
        x = self.linear(x)  # shape [B, T, n_embed]
        x = self.swish(x)  # Apply Swish activation
        x = self.linear_out(x)
        x = self.dropout(x, training=self.training)  # Apply dropout if in training mode
        return x


class Block(keras.layers.Layer):
    """
    A single block of the GPT model, consisting of self-attention and feed-forward layers.
    """
    def __init__(self, n_embed, num_heads, training=True):
        super().__init__()
        self.sa = MultiHeadAttention(n_heads=num_heads, head_size=n_embed // num_heads, training=training)  # 4 heads, each of size n_embed // 4
        self.ff = FeedForward(n_embed=n_embed, training=training)
        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-5)  # Layer normalization after self-attention
        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-5)  # Layer normalization after feed-forward

    def call(self, x: tf.Tensor):
        # See if weighted sum would help here where the weights are leanable
        # This might help since a naive addition splits the gradients equally across the two branches.
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class Blocks(keras.layers.Layer):
    """
    A stack of multiple blocks of self-attention and feed-forward layers.
    """
    def __init__(self, n_embed, num_heads, n_blocks, training=True):
        super().__init__()
        self.blocks = [Block(n_embed=n_embed, num_heads=num_heads, training=training) for _ in range(n_blocks)]
        self.ln = keras.layers.LayerNormalization(epsilon=1e-5)  # Layer normalization after all blocks

    def call(self, x: tf.Tensor):
        for block in self.blocks:
            x = block(x)
        return self.ln(x)

# --- Model Definition ---

class GPT(keras.Model):
    def __init__(self, training=True):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table.
        self.embedding = keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=N_EMBED)
        self.pos_embedding = keras.layers.Embedding(input_dim=BLOCK_SIZE, output_dim=N_EMBED)
        self.num_heads = 8  # Number of attention heads
        assert N_EMBED % self.num_heads == 0, "N_EMBED must be divisible by num_heads"
        self.head_size = N_EMBED // self.num_heads  # Size of each attention head
        self.transformer_blocks = Blocks(n_embed=N_EMBED, num_heads=self.num_heads, n_blocks=4, training=training)
        self.lm_head = keras.layers.Dense(units=VOCAB_SIZE)  # No bias for logits

    def build(self, input_shape):
        """
        Explicitly builds the model's layers. This method is called by Keras the first
        time the model is used, and defining it helps prevent warnings about unbuilt state.
        """
        # The sub-layers will be built automatically when called.
        super().build(input_shape)
    
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
        B, T = idx.shape  # B is batch size, T is block size
        # idx and targets are both tensors of shape [batch_size, block_size]
        token_embeddings = self.embedding(idx) # shape [batch_size, block_size, n_embed]
        pos_embeddings = self.pos_embedding(tf.range(0, T)) # shape [block_size, n_embed]
        x = token_embeddings + pos_embeddings  # Add positional embeddings
        x = self.transformer_blocks(x) # Apply one blocks of transformer layers
        logits = self.lm_head(x) # shape [batch_size, block_size, vocab_size]

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
        for _ in tqdm(range(max_new_tokens)):
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
    model = GPT(training=True)

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
    bar = tqdm(range(NUM_ITERATIONS))
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
        if step % EVAL_INTERVAL == 0 or step == NUM_ITERATIONS - 1:
            x_val_batch, y_val_batch = get_batch("val")
            _, val_loss = model(x_val_batch, y_val_batch)
        
        bar.set_postfix(train_loss=loss.numpy(), val_loss=val_loss.numpy())

    print(f"\nTraining finished. Final training loss: {loss.numpy():.4f}")
    print(f"Final validation loss: {val_loss.numpy():.4f}")
    print("-" * 25)
    # Save the trained model
    model.save_weights("gpt_model.weights.h5")
    del model

    # --- Generate text from the TRAINED model ---
    print("\n--- Generating text from TRAINED model ---")
    model = GPT(training=False)
    x_val_batch, y_val_batch = get_batch("val")
    _, _ = model(x_val_batch, y_val_batch)  # Run a forward pass to build the model
    model.load_weights("gpt_model.weights.h5")
    context_trained = tf.zeros((1, 1), dtype=tf.int64) 
    generated_output_trained = model.generate(idx=context_trained, max_new_tokens=500)
    decoded_text_trained = decode(generated_output_trained[0].numpy().tolist())
    print(decoded_text_trained)
    print("\nScript finished.")

if __name__ == "__main__":
    main()