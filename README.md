# LLM Comparison Hub

A comprehensive tool for comparing responses from multiple Large Language Models (GPT-4, Claude 3, Gemini 1.5) with built-in evaluation, analysis, and visualization capabilities.

## Features

- **Multi-Model Response Generation**: Generate responses from GPT-4, Claude 3, and Gemini 1.5
- **Comprehensive Evaluation**: Round-robin evaluation where each model evaluates all others
- **Real-time Query Detection**: Automatically detect and enhance real-time queries with Google search
- **Data Analysis & Visualization**: Generate charts and analysis reports
- **Modular Architecture**: Clean, production-ready code with separated concerns
- **Gradio Web Interface**: User-friendly web UI for all features

## Project Structure

### Core Modules

- **`gradio_full_llm_eval.py`** - Main Gradio web interface (UI orchestration only)
- **`response_generator.py`** - Handles all LLM response generation
- **`round_robin_evaluator.py`** - Handles comprehensive model evaluation
- **`realtime_detector.py`** - Detects real-time queries
- **`search_fallback.py`** - Integrates Google search for real-time information
- **`llm_prompt_eval_analysis.py`** - Data analysis and visualization

### Configuration Files

- **`requirements.txt`** - Python dependencies
- **`.env`** - API keys and configuration (create this file)
- **`README.md`** - This documentation

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd LLM-Compare-Hub
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**:
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
python gradio_full_llm_eval.py
```

The interface provides:
- **Input Section**: Enter prompts and configure options
- **Results Tabs**: View responses, evaluations, search results, and visualizations
- **API Status**: Check which API keys are available
- **Export**: Save results to files

### Standalone Tools

Each module can be used independently:

#### Response Generator
```bash
python response_generator.py
```
- Interactive mode for single prompts
- Batch mode for multiple prompts from file

#### Round-Robin Evaluator
```bash
python round_robin_evaluator.py
```
- Test the evaluation system
- View evaluation metrics

#### Analysis Tool
```bash
python llm_prompt_eval_analysis.py
```
- Analyze latest CSV results
- Generate visualizations
- Create comprehensive reports

#### Response Logger
```bash
python llm_response_logger.py
```
- Quick testing of all models
- Batch testing from files

## API Requirements

### Required APIs
- **OpenAI API**: For GPT-4 responses
- **Anthropic API**: For Claude 3 responses
- **Google Generative AI**: For Gemini 1.5 responses

### Optional APIs
- **Google Custom Search**: For real-time query enhancement

## Evaluation Metrics

The system evaluates responses on multiple criteria:
- **Helpfulness**: How useful and informative is the response?
- **Correctness**: How accurate and factually correct is the response?
- **Coherence**: How well-structured and logical is the response?
- **Tone**: How appropriate and professional is the tone?
- **Accuracy**: How precise and detailed is the information?
- **Relevance**: How well does the response address the prompt?
- **Completeness**: How comprehensive is the response?
- **Clarity**: How clear and easy to understand is the response?

## Output Files

### Results Directory
- **CSV Files**: Comprehensive evaluation results with timestamps
- **Analysis Reports**: Detailed analysis and insights
- **Visualizations**: Charts and graphs in PNG format
- **Export Files**: Text exports of complete results

### File Naming Convention
- `comprehensive_eval_YYYYMMDD_HHMMSS.csv` - Evaluation results
- `evaluation_report_YYYYMMDD_HHMMSS.txt` - Analysis reports
- `export_YYYYMMDD_HHMMSS.txt` - Exported results

## Modular Architecture

### Design Principles
- **Separation of Concerns**: Each file has a specific responsibility
- **Clean Code**: No emojis or decorative symbols
- **Production Ready**: Error handling and logging throughout
- **Reusable Components**: Modules can be used independently
- **Configurable**: Easy to modify and extend

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `gradio_full_llm_eval.py` | UI orchestration and display |
| `response_generator.py` | LLM API calls and response collection |
| `round_robin_evaluator.py` | Model evaluation and scoring |
| `realtime_detector.py` | Real-time query detection |
| `search_fallback.py` | Google search integration |
| `llm_prompt_eval_analysis.py` | Data analysis and visualization |

## Error Handling

The system includes comprehensive error handling:
- **API Failures**: Graceful handling of API errors
- **Missing Keys**: Clear indication of missing API keys
- **Network Issues**: Retry logic and fallback options
- **Data Validation**: Input validation and sanitization

## Contributing

1. Follow the modular architecture
2. Maintain clean, production-ready code
3. Add proper error handling
4. Update documentation for new features
5. Test all modules independently

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the API key configuration
2. Verify all dependencies are installed
3. Review the error messages in the console
4. Check the results directory for output files 