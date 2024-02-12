# LlamaIndex Project README

## Overview

The LlamaIndex Project leverages the LlamaIndex library to fetch, index, and query web pages. It demonstrates how to programmatically read web content, convert it into a searchable index, and perform natural language queries on this index. This README provides a detailed guide on setting up and using the project, tailored for environments that utilize `pipenv` and Python 3.10.

## Features

- **Secure Environment Variable Management**: Uses `dotenv` for managing environment variables safely.
- **Efficient Web Page Indexing**: Automates the process of reading web pages, converting them to text, and indexing the content.
- **Advanced Query Capabilities**: Employs a query engine for executing natural language searches on indexed content.

## Prerequisites

Before you get started, ensure the following requirements are met:
- Python 3.10
- pipenv for Python package and virtual environment management

## Installation

1. **Clone the Project Repository**
   - Use Git to clone the repository to your local machine.

2. **Install Dependencies with Pipenv**
   - In the project's root directory, run the following command to set up a virtual environment with Python 3.10 and install the required packages:
     ```bash
     pipenv --python 3.10 install
     ```

3. **Set Up Environment Variables**
   - Populate the `.env` file with your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Usage

Follow these steps to run the LlamaIndex Project:

1. **Activate the Virtual Environment**
   - Before running the script, activate the pipenv shell to ensure you're using the project's virtual environment:
     ```bash
     pipenv shell
     ```

2. **Run the Script**
   - Start the main script via the command line:
     ```bash
     pipenv run python main.py
     ```
   - The script will automatically read the specified web page, index the content, and perform a query with the text "What is LlamaIndex".

3. **View Query Results**
   - The output will display the query results in the console, showcasing the project's ability to index and search web content efficiently.

## Troubleshooting

- **Missing OpenAI API Key**: If an error indicates that the `OPENAI_API_KEY` environment variable is not set, double-check the `.env` file for correct setup.
- **Dependency Issues**: Should there be any problems with missing packages, ensure all dependencies are correctly installed by running `pipenv install` once more.

## License

This project is distributed under the MIT License. For more details, see the LICENSE file in the project repository.

## Acknowledgments

- A special thank you to the LlamaIndex library authors for providing the foundational tools used in this project.
- This project makes use of the OpenAI API for generating text embeddings and facilitating queries.
