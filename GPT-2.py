import tensorflow as tf
import keras


class GPTConfig:
    """Configuration for the GPT model."""
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dtype: tf.DType = tf.float32


class CasualSelfAttention(keras.layers.Layer):
    """Causal self-attention layer."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0, \
            "n_embed must be divisible by n_head"
        self.config = config
        self.c_attn = keras.layers.Dense(config.n_embed * 3, dtype=config.dtype)
        self.c_proj = keras.layers.Dense(config.n_embed, dtype=config.dtype)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        tril_const = tf.linalg.band_part(
            tf.ones((config.block_size, config.block_size), dtype=config.dtype),
            -1, 0
        )
        self.tril = tf.Variable(tril_const, trainable=False)

    def call(self, x: tf.Tensor, training=False):
        """Forward pass for causal self-attention."""
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = tf.split(qkv, 3, axis=-1)
        q = tf.reshape(q, (B, T, self.n_head, C // self.n_head))
        k = tf.reshape(k, (B, T, self.n_head, C // self.n_head))
        v = tf.reshape(v, (B, T, self.n_head, C // self.n_head))
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 3, 1])
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        weights = q @ k
        weights = weights / tf.sqrt(tf.cast(C // self.n_head, self.config.dtype))
        weights = tf.where(self.tril[:T, :T] == 0,
                           tf.constant(float("-inf"), dtype=self.config.dtype), weights)
        weights = tf.nn.softmax(weights, axis=-1)
        y = weights @ v
        y = tf.transpose(y, perm=[0, 2, 1, 3])
        y = tf.reshape(y, (B, T, C))
        y = self.c_proj(y)
        return y


class MLP(keras.layers.Layer):
    """Multi-layer perceptron layer."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = keras.layers.Dense(config.n_embed * 4, activation='gelu', dtype=config.dtype)
        self.c_proj = keras.layers.Dense(config.n_embed, dtype=config.dtype)
        self.dropout = keras.layers.Dropout(0.1)

    def call(self, x: tf.Tensor, training=False):
        """Forward pass for the MLP."""
        x = self.c_fc(x)
        x = self.dropout(x, training=training)
        x = self.c_proj(x)
        return x


class TransformerBlock(keras.layers.Layer):
    """Transformer block layer."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = keras.layers.LayerNormalization(dtype=config.dtype)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = keras.layers.LayerNormalization(dtype=config.dtype)
        self.mlp = MLP(config)  # Sometimes called FFN

    def call(self, x: tf.Tensor, training=False):
        """Forward pass for the transformer block."""
        x = x + self.attn(self.ln_1(x), training=training)
        x = x + self.mlp(self.ln_2(x), training=training)
        return x


class GPTModel(keras.Model):
    """GPT model."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = {
            "wte": keras.layers.Embedding(config.vocab_size, config.n_embed, dtype=config.dtype),
            "wpe": keras.layers.Embedding(config.block_size, config.n_embed, dtype=config.dtype),
            "h": [
                TransformerBlock(config) for _ in range(config.n_layer)
            ],
            "ln_f": keras.layers.LayerNormalization(dtype=config.dtype)
        }
        # The lm_head is not a separate layer. Instead, we will do a matrix
        # multiplication using the transposed token embedding weights.
        self.lm_head = self.transformer["wte"]
 
    def call(self, idx: tf.Tensor, targets: tf.Tensor=None, training: bool=False):
        """call pass for the GPT model."""
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Cannot call sequence of length {T}, block size is {self.config.block_size}"

        pos = tf.range(0, T, dtype=tf.int32) # (T, )
        pos_emb = self.transformer["wpe"](pos) # (T, C)
        token_emb = self.transformer["wte"](idx) # (B, T, C)
        x = token_emb + pos_emb # (B, T, C)

        for block in self.transformer["h"]:
            x = block(x, training=training)
        
        x = self.transformer["ln_f"](x) # (B, T, C)
        # In GPT-2, the output projection weights are shared with the
        # input token embedding weights.
        # We use the transposed embedding matrix for the final linear layer.
        logits = tf.matmul(x, self.lm_head.embeddings, transpose_b=True) # (B, T, vocab_size)

        if targets is not None:
            # Keras's sparse_categorical_crossentropy expects logits of shape
            # [batch_size, sequence_length, num_classes] and targets of shape
            # [batch_size, sequence_length], which matches our data.
            loss = keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
            return logits, loss

        return logits

import tiktoken
class Dataloader:
    """Dataloader for the GPT model."""
    def __init__(self, batch_size: int, block_size: int):
        self.batch_size = batch_size
        self.block_size = block_size

        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        # self.tokens = tf.convert_to_tensor(tokens, dtype=tf.int32)
        self.tokens = tf.convert_to_tensor(tokens, dtype=tf.bfloat16)
        print(f"Loaded {len(tokens)} tokens from input.txt")
        print("1 epoch = ", len(tokens) // (batch_size * block_size), "batches")

        # state
        self.current_position = 0

    def next_batch(self):
        """Get the next batch of data."""
        tokens = self.tokens
        B, T = self.batch_size, self.block_size
        buff = self.tokens[self.current_position: self.current_position + ((B * T) + 1)]
        x = tf.reshape(buff[:-1], (B, T))  # (B, T)
        y = tf.reshape(buff[1:], (B, T))  # (B, T)
        self.current_position += (B * T)
        if self.current_position + (B * T) >= len(tokens):
            self.current_position = 0
        return x, y

encoder = tiktoken.get_encoding("gpt2")
model = GPTModel(GPTConfig())
optimizer = keras.optimizers.AdamW(learning_rate=3e-4)

with open("input.txt", "r") as f:
    text = f.read()

dataloader = Dataloader(batch_size=4, block_size=32)

x, y = dataloader.next_batch()
logits, loss = model(x, y)
print("loss: ", loss.numpy())

for _ in range(50):
    x, y = dataloader.next_batch()
    with tf.GradientTape() as tape:
        logits, loss = model(x, y, training=True)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("loss: ", loss.numpy())


# num_sequences = 5
# x = tf.convert_to_tensor([tokens for _ in range(num_sequences)], dtype=tf.int32)
# max_length = 30

# logits, loss = model(x, y)
# print("loss: ", loss.numpy())

# while tf.shape(x)[1] < max_length:
#     logits = model(x) # (B, T, vocab_size)
#     logits = logits[:, -1, :]  # (B, vocab_size)
    
#     probs = tf.nn.softmax(logits, axis=-1)  # (B, vocab_size)
#     topk_values, topk_indices = tf.math.top_k(probs, k=50)  # (B, k)
#     ix = tf.random.categorical(tf.math.log(topk_values), num_samples=1)  # (B, 1)
#     xcol = tf.gather(topk_indices, ix, batch_dims=1)  # (B, 1)
#     x = tf.concat([x, xcol], axis=1)  # (B, T + 1)

# for i in range(num_sequences):
#     tokens = x[i, :].numpy().tolist()
#     text = encoder.decode(tokens)
#     print(">", text)
