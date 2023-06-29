from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "torch",
    "transformers",
    "evaluate",
    "datasets",
    "tqdm",
    "accelerate",
    "bitsandbytes",
    "ftfy",
    "regex",
    "omegaconf",
    "openai",
    "opencv-python",
    "nltk",
    "scikit-learn",
    "open_clip_torch",
    "wandb",
    "fairscale",
    "huggingface_hub",
]

setup(
    name="llm-lens",
    version="0.0.0.3",
    description=(
        "llm-lens is a Python package for CV as NLP, "
        "where you can run very descriptive image modules "
        "on images, and then pass those descriptions to a "
        "Large Language Model (LLM) to reason about those "
        "images."
    ),
    author="The Contextual AI Tech Team",
    author_email="tech@contextual.ai",
    url="https://github.com/ContextualAI/lens",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
