defmodule SimpleEmbedder.MixProject do
  use Mix.Project

  def project do
    [
      app: :simple_embedder,
      version: "0.2.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      package: %{
        description: "A package that wraps popular python-based embedding generators.",
        links: %{source: "https://github.com/lambadalambda/simple_embedder"},
        source_url: "https://github.com/lambadalambda/simple_embedder",
        name: "simple_embedder",
        licenses: ["CC0-1.0"],
        exclude_patterns: ["__pycache__"]
      },
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies. 
  defp deps do
    [
      {:erlport, "~> 0.11.0"},
      {:mix_test_watch, "~> 1.0", only: [:dev, :test], runtime: false},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
    ]
  end
end
