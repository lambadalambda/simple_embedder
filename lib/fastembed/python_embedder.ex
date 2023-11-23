defmodule SimpleEmbedder.FastEmbed.PythonEmbedder do
  use GenServer

  @behaviour SimpleEmbedder.EmbedderAPI

  @supported_models %{
    "BAAI/bge-small-en-v1.5" => %{
      name: "BAAI/bge-small-en-v1.5",
      vector_size: 384
    }
  }

  @supported_model_names @supported_models |> Map.keys()
  @default_model "BAAI/bge-small-en-v1.5"

  @impl true
  def init(opts \\ []) do
    model = Keyword.get(opts, :model, @default_model)

    case model in @supported_model_names do
      true -> {:ok, %{model_name: model}, {:continue, :load_model}}
      false -> {:stop, :error, "Invalid model name: #{model}"}
    end
  end

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @impl true
  def get_text_embedding(pid \\ __MODULE__, text) do
    GenServer.call(pid, {:get_text_embedding, text})
  end

  @impl true
  def handle_continue(:load_model, state) do
    python_path = Path.join([__DIR__, "python_embedder"])

    {:ok, python} =
      :python.start(
        python_path: python_path |> String.to_charlist()
        # python: Path.join(:python_path, "venv/bin/python") |> String.to_charlist()
      )

    res = :python.call(python, :python_embedder, :load_model, [state.model_name])

    case res do
      :ok -> {:noreply, Map.put(state, :python, python)}
      :error -> {:stop, :error, state}
    end
  end

  @impl true
  def handle_call({:get_text_embedding, text}, _from, state) do
    embedding = :python.call(state.python, :python_embedder, :get_text_embedding, [text])
    {:reply, {:ok, embedding}, state}
  end
end
