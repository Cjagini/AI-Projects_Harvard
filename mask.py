import sys
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained BERT model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)

    if mask_token_index is None:
        sys.exit("Input must include mask token.")

    # Use BERT to predict masked token
    model = TFBertForMaskedLM.from_pretrained(MODEL, from_pt=True)
    result = model(inputs, output_attentions=True)
    mask_token_logits = result.logits[0, mask_token_index]

    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified mask_token_id.
    """
    # inputs['input_ids'] is a tensor of shape [1, sequence_length]
    input_ids = inputs["input_ids"][0].numpy()
    for index, token_id in enumerate(input_ids):
        if token_id == mask_token_id:
            return index
    return None


def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing an RGB color for the
    given attention_score.
    """
    # Scale 0-1 to 0-255 linearly
    color_val = int(attention_score * 255)
    return (color_val, color_val, color_val)


def visualize_attentions(tokens, attentions):
    """
    Produce diagrams for all attention heads.
    """
    # attentions is a tuple of 12 layers
    # Each layer is a tensor of shape [batch, heads, seq_len, seq_len]
    for layer_idx, layer in enumerate(attentions):
        # Number of heads is usually 12
        num_heads = layer.shape[1]
        for head_idx in range(num_heads):
            # Extract scores for this specific head
            # j=0 (beam/batch index), k=head_idx
            head_attentions = layer[0][head_idx]

            generate_diagram(
                layer_idx + 1,
                head_idx + 1,
                tokens,
                head_attentions
            )


def generate_diagram(layer_number, head_number, tokens, attentions):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each token,
    and cells are colored based on attention scores.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGB", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the setting
    for i, token in enumerate(tokens):
        # Draw tokens at top
        token_image = Image.new("RGB", (PIXELS_PER_WORD, GRID_SIZE), "black")
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text((0, 0), token, fill="white", font=FONT)
        token_image = token_image.rotate(90, expand=True)
        img.paste(token_image, (PIXELS_PER_WORD + i * GRID_SIZE, 0))

        # Draw tokens at left
        draw.text((0, PIXELS_PER_WORD + i * GRID_SIZE),
                  token, fill="white", font=FONT)

    # Draw each cell in the visualizer
    for i in range(len(tokens)):
        for j in range(len(tokens)):

            # Get color based on attention
            color = get_color_for_attention_score(attentions[i][j])

            # Draw a rectangle for the cell
            draw.rectangle(
                [
                    (PIXELS_PER_WORD + j * GRID_SIZE,
                     PIXELS_PER_WORD + i * GRID_SIZE),
                    (PIXELS_PER_WORD + (j + 1) * GRID_SIZE,
                     PIXELS_PER_WORD + (i + 1) * GRID_SIZE)
                ],
                fill=color
            )

    # Save image
    img.save(f"Attention_Layer_{layer_number}_Head_{head_number}.png")


if __name__ == "__main__":
    main()
