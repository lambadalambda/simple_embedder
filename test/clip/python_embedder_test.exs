defmodule SimpleEmbedder.CLIP.PythonEmbedderTest do
  use ExUnit.Case

  alias SimpleEmbedder.CLIP.PythonEmbedder

  setup_all do
    {:ok, pid} = PythonEmbedder.start_link(name: nil)

    # To ensure we're done loading when running the tests
    :pong = GenServer.call(pid, :ping)

    on_exit(fn ->
      GenServer.stop(pid)
    end)

    %{pid: pid}
  end

  test "by default, uses a fast openai clip model", %{pid: pid} do
    assert %{name: "openai/clip-vit-base-patch32", vector_size: 512} ==
             PythonEmbedder.current_model(pid)
  end

  test "it generates a text embedding", %{pid: pid} do
    assert {:ok, embedding} = PythonEmbedder.get_text_embedding(pid, "a cool text")
    assert [-0.13147132098674774, 0.4853159487247467, -0.017824381589889526 | _] = embedding
  end

  test "it generates an image embedding", %{pid: pid} do
    expected = [0.4964962899684906, 0.5463931560516357, -0.36338984966278076]

    {:ok, embedding} =
      PythonEmbedder.get_image_embedding(pid, Path.absname("test/fixtures/dolly.png"))

    assert expected == embedding |> Enum.take(3)
  end

  test "if started with an invalid model name, returns an error" do
    {:error, _} = PythonEmbedder.start_link(name: nil, model: "invalid")
  end
end
