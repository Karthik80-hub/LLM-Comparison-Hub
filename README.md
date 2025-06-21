---
title: LLM Comparison Hub
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.34.2"
app_file: app.py
pinned: false
---

# LLM Comparison Hub

A comprehensive tool for comparing responses from multiple Large Language Models (GPT-4, Claude 3, Gemini 1.5) with built-in evaluation, analysis, and visualization capabilities.

**Live Demo:** [https://huggingface.co/spaces/chunchu-08/LLM-Comparison-Hub](https://huggingface.co/spaces/chunchu-08/LLM-Comparison-Hub)

## Overview

This application provides a complete LLM comparison and evaluation system that generates responses from multiple models, performs round-robin evaluations where each model evaluates all others, and provides comprehensive analysis with interactive visualizations.

## Key Features

- **Multi-Model Response Generation**: Generate responses from GPT-4, Claude 3, and Gemini 1.5
- **Round-Robin Evaluation System**: Each model evaluates all other models for comprehensive comparison
- **Real-time Query Detection**: Automatically detect and enhance real-time queries with Google search
- **ATS Scoring**: Resume vs Job Description matching with detailed feedback
- **Interactive Data Analysis & Visualization**: Generate interactive charts, heatmaps, and performance reports
- **Batch Processing**: Handle multiple prompts from CSV files
- **Modular Architecture**: Clean, production-ready code with separated concerns
- **Gradio Web Interface**: User-friendly web UI for all features
- **Export Capabilities**: ZIP bundles with all results and interactive visualizations
- **Automated Deployment**: GitHub Actions for continuous deployment to Hugging Face Spaces

## Project Architecture

### Core Application Files

- **`app.py`** - Main Gradio web interface (UI orchestration and deployment)
- **`response_generator.py`** - Handles all LLM response generation and comparison
- **`round_robin_evaluator.py`** - Comprehensive model evaluation system
- **`llm_prompt_eval_analysis.py`** - Data analysis and visualization engine
- **`llm_response_logger.py`** - Quick testing and logging tool

### Supporting Modules

- **`realtime_detector.py`** - Detects real-time queries that need current information
- **`search_fallback.py`** - Integrates Google search for real-time information enhancement

### Configuration Files

- **`requirements.txt`** - Python dependencies and versions
- **`.env`** - API keys and configuration (create this file)
- **`.github/workflows/deploy-to-hf.yml`** - GitHub Actions for automated deployment

## Installation

### Prerequisites

- Python 3.8 or higher
- API keys for OpenAI, Anthropic, and Google Generative AI

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd LLM-Compare-Hub
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**:
   Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   CLAUDE_API_KEY=your_claude_key_here
   GEMINI_API_KEY=your_gemini_key_here
   GOOGLE_API_KEY=your_google_key_here
   GOOGLE_CSE_ID=your_google_cse_id_here
   ```

## Usage

### Web Interface (Recommended)

Launch the Gradio web interface:
```bash
python app.py
```

The interface provides:
- **Input Section**: Enter prompts, upload files, and configure options
- **Results Tabs**: View responses, evaluations, search results, and interactive visualizations
- **Export Options**: Download results as ZIP bundles with interactive HTML charts
- **Real-time Features**: Automatic query detection and search enhancement

### Standalone Tools

Each module can be used independently for specific tasks:

#### Response Generator
```bash
python response_generator.py
```
- Interactive mode for single prompts
- Batch mode for multiple prompts from file
- Side-by-side response comparison

#### Round-Robin Evaluator
```bash
python round_robin_evaluator.py
```
- Test the evaluation system
- View evaluation metrics and scores
- Export results to CSV

#### Analysis Tool
```bash
python llm_prompt_eval_analysis.py
```
- Analyze latest CSV results
- Generate visualizations and charts
- Create comprehensive performance reports

#### Response Logger
```bash
python llm_response_logger.py
```
- Quick testing of all models
- Batch testing from files
- Rapid evaluation and logging

## API Requirements

### Required APIs
- **OpenAI API**: For GPT-4 responses and ATS scoring
- **Anthropic API**: For Claude 3 responses
- **Google Generative AI**: For Gemini 1.5 responses

### Optional APIs
- **Google Custom Search**: For real-time query enhancement

## Evaluation Metrics

The system evaluates responses on eight comprehensive criteria:

- **Helpfulness**: How useful and informative is the response?
- **Correctness**: How accurate and factually correct is the response?
- **Coherence**: How well-structured and logical is the response?
- **Tone Score**: How appropriate and professional is the tone?
- **Accuracy**: How precise and detailed is the information?
- **Relevance**: How well does the response address the prompt?
- **Completeness**: How comprehensive is the response?
- **Clarity**: How clear and easy to understand is the response?

## ATS Scoring System

When a resume and job description are provided, the system performs ATS (Applicant Tracking System) scoring:

- **Keyword Matching**: Identifies relevant skills and qualifications
- **Section Weighting**: Prioritizes important sections
- **Semantic Similarity**: Analyzes meaning and context
- **Recency/Frequency**: Considers experience relevance
- **Penalty Detection**: Identifies potential issues
- **Aggregation**: Provides overall match score

## Output and Results

### Generated Files
- **CSV Files**: Comprehensive evaluation results with timestamps
- **Analysis Reports**: Detailed analysis and insights
- **Interactive Visualizations**: Interactive HTML charts and graphs
- **Export Bundles**: ZIP files containing all results and interactive charts

### File Naming Convention
- `evaluation_YYYYMMDD_HHMMSS.csv` - Evaluation results
- `batch_YYYYMMDD_HHMMSS/` - Results directory
- `heatmap.html`, `radar.html`, `barchart.html` - Interactive visualization files
- `bundle.zip` - Complete export package

## Technical Architecture

### Design Principles
- **Separation of Concerns**: Each file has a specific responsibility
- **Clean Code**: Production-ready without decorative elements
- **Error Handling**: Comprehensive error handling and logging
- **Reusable Components**: Modules can be used independently
- **Configurable**: Easy to modify and extend
- **Hugging Face Compatible**: No external browser dependencies for chart generation

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `app.py` | UI orchestration and deployment |
| `response_generator.py` | LLM API calls and response collection |
| `round_robin_evaluator.py` | Model evaluation and scoring |
| `realtime_detector.py` | Real-time query detection |
| `search_fallback.py` | Google search integration |
| `llm_prompt_eval_analysis.py` | Data analysis and visualization |

## Deployment

### Automated Deployment with GitHub Actions

The project includes automated deployment to Hugging Face Spaces using GitHub Actions:

#### Setup Requirements

1. **Hugging Face Access Token**:
   - Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Create a new token with **Write** permissions
   - Copy the token (starts with `hf_...`)

2. **GitHub Repository Secrets**:
   - Go to your GitHub repository Settings
   - Navigate to Secrets and variables → Actions
   - Add a new repository secret:
     - **Name**: `HF_TOKEN`
     - **Value**: Your Hugging Face token

#### Deployment Workflow

The `.github/workflows/deploy-to-hf.yml` file automatically:
- Triggers on pushes to the main branch
- Deploys changes to Hugging Face Spaces
- Maintains continuous integration

#### Usage

After setup, simply push to GitHub:
```bash
git add .
git commit -m "Update application"
git push origin main
```

The GitHub Action will automatically deploy to Hugging Face Spaces.

### Manual Deployment

For local deployment, ensure all dependencies are installed and API keys are configured.

## Error Handling

The system includes comprehensive error handling:
- **API Failures**: Graceful handling of API errors with fallback options
- **Missing Keys**: Clear indication of missing API keys
- **Network Issues**: Retry logic and connection management
- **Data Validation**: Input validation and sanitization
- **File Processing**: Robust handling of various file formats

## Development and Testing

### Testing Tools
- **`test_standalone_tools.py`**: Demonstrates usage of all standalone tools
- **Batch Testing**: Process multiple prompts efficiently
- **Performance Monitoring**: Track evaluation metrics over time

### Development Guidelines
1. Follow the modular architecture
2. Maintain clean, production-ready code
3. Add proper error handling
4. Update documentation for new features
5. Test all modules independently

## Contributing

1. Follow the established modular architecture
2. Maintain clean, production-ready code standards
3. Add comprehensive error handling
4. Update documentation for any new features
5. Test all modules independently before submission

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the API key configuration in `.env`
2. Verify all dependencies are installed correctly
3. Review error messages in the console output
4. Check the results directory for output files
5. Consult the project documentation for detailed module descriptions

## Live Application

Access the live application at: [https://huggingface.co/spaces/chunchu-08/LLM-Comparison-Hub](https://huggingface.co/spaces/chunchu-08/LLM-Comparison-Hub) "<!-- trigger deploy -->" 
