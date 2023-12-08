defmodule SimpleEmbedder.FastEmbed.PythonEmbedder do
  use GenServer

  @behaviour SimpleEmbedder.EmbedderAPI
  alias SimpleEmbedder.EmbedderAPI

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
    python_exec_path = Keyword.get(opts, :python_exec_path)

    case model in @supported_model_names do
      true ->
        {:ok, %{model_name: model, python_exec_path: python_exec_path}, {:continue, :load_model}}

      false ->
        {:stop, :error, "Invalid model name: #{model}"}
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
  def get_image_embedding(_) do
    raise "Not implemented for this model"
  end

  @impl true
  def handle_continue(:load_model, state) do
    python_path = Path.join([__DIR__, "python_embedder"])

    {:ok, python} = EmbedderAPI.start_python(python_path, state.python_exec_path)

    res = :python.call(python, :python_embedder, :load_model, [state.model_name])

    case res do
      :ok -> {:noreply, Map.put(state, :python, python)}
      {:error, error} -> {:stop, :error, Map.put(state, :error, error)}
    end
  end

  @impl true
  def handle_call({:get_text_embedding, text}, _from, state) do
    embedding = :python.call(state.python, :python_embedder, :get_text_embedding, [text])
    {:reply, {:ok, embedding}, state}
  end

  @impl true
  def terminate(_reason, state) do
    if state[:python] do
      :python.stop(state.python)
    end

    :ok
  end
end
