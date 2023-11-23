defmodule SimpleEmbedder.FastEmbed.PythonEmbedderTest do
  use ExUnit.Case
  alias SimpleEmbedder.FastEmbed.PythonEmbedder

  setup_all do
    {:ok, pid} = PythonEmbedder.start_link(name: nil)

    on_exit(fn ->
      GenServer.stop(pid)
    end)

    %{pid: pid}
  end

  test "it generates a text embedding", %{pid: pid} do
    assert {:ok, embedding} = PythonEmbedder.get_text_embedding(pid, "a cool text")

    assert [-0.06423746794462204, 0.021873118355870247, 0.03726368024945259] ==
             embedding |> Enum.take(3)
  end
end
