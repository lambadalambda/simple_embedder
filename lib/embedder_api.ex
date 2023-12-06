defmodule SimpleEmbedder.EmbedderAPI do
  @callback get_text_embedding(String.t()) :: {:ok, [number()]}
  @callback get_image_embedding(String.t()) :: {:ok, [number()]}

  def start_python(python_path, python_exec_path) do
    if python_exec_path do
      :python.start(
        python_path: python_path |> String.to_charlist(),
        python: python_exec_path |> String.to_charlist()
      )
    else
      :python.start(python_path: python_path |> String.to_charlist())
    end
  end
end
