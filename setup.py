from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gateway-x",
    version="32.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-engine AI consensus system with statistical rigor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gateway-x-consensus",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
    ],
    extras_require={
        "anthropic": ["anthropic>=0.7.0"],
        "openai": ["openai>=1.3.0"],
        "google": ["google-generativeai>=0.3.0"],
        "cohere": ["cohere>=4.37"],
        "all": [
            "anthropic>=0.7.0",
            "openai>=1.3.0",
            "google-generativeai>=0.3.0",
            "cohere>=4.37",
        ],
    },
    entry_points={
        "console_scripts": [
            "gateway-x=gatewayx.server:main",
        ],
    },
)
