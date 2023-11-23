defmodule SimpleEmbedder.EmbedderAPI do
  @callback get_text_embedding(String.t()) :: {:ok, [number()]}
  @callback get_image_embedding(String.t()) :: {:ok, [number()]}
end
