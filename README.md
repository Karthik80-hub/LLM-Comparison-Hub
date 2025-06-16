# LLM-Compare-Hub

A comprehensive tool for evaluating and comparing responses from multiple AI language models (GPT-4, Claude 3, and Gemini 1.5) with real-time search integration and advanced analytics.

## Features

### 1. Multi-Model Evaluation
- **Supported Models**:
  - GPT-4 (OpenAI)
  - Claude 3 Opus (Anthropic)
  - Gemini 1.5 Pro (Google)
- **Round-Robin Evaluation**: Each model's response is evaluated by another model in a cycle
- **Comprehensive Metrics**:
  - Helpfulness
  - Correctness
  - Coherence
  - Tone Score
  - Accuracy
  - Relevance
  - Completeness
  - Clarity

### 2. Real-Time Information Integration
- **Automatic Detection**: Identifies prompts requiring real-time information
- **Google Search Integration**: Fetches relevant search results for real-time queries
- **Enhanced Responses**: Models incorporate search results into their responses
- **Transparent Reasoning**: Models explain how search results influenced their responses

### 3. Advanced Analytics & Visualization
- **Interactive Dashboard**: Gradio-based user interface
- **Visualization Tools**:
  - Correlation Heatmap: Shows relationships between metrics
  - Bar Chart: Compares average scores across models
  - Radar Chart: Displays metric distribution for each model
- **Customizable Controls**:
  - Correlation Threshold: Filter metric relationships
  - Metric Weight: Adjust importance of metrics

### 4. Comprehensive Logging
- **Detailed CSV Export**:
  - Timestamp of evaluation
  - Original prompt
  - Model responses
  - Evaluation metrics
  - Reasoning and notes
  - Round-robin evaluation results
- **Automatic File Management**:
  - Results stored in `results/` directory
  - Files named `ai_prompt_eval_YYYYMMDD_HHMMSS.csv`
  - Easy to track and compare evaluations

## Setup

1. **Clone the Repository**:
   ```bash
   git clone [repository-url]
   cd LLM-Compare-Hub
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   Create a `.env` file with the following API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   CLAUDE_API_KEY=your_claude_key
   GEMINI_API_KEY=your_gemini_key
   GOOGLE_API_KEY=your_google_key
   GOOGLE_CSE_ID=your_custom_search_engine_id
   ```

## Usage

1. **Start the Application**:
   ```bash
   python gradio_full_llm_eval.py
   ```

2. **Using the Dashboard**:
   - Enter your prompt in the text box
   - Click "Evaluate Prompt" to process
   - View responses and metrics for each model
   - Adjust visualization controls as needed
   - Download results as CSV

3. **Understanding the Results**:
   - **Response Display**: Shows each model's response with metrics
   - **Metrics Panel**: Displays detailed evaluation scores
   - **Visualizations**: Interactive charts for metric analysis
   - **CSV Export**: Complete evaluation data for further analysis

## Features in Detail

### Real-Time Query Handling
- The system automatically detects if a prompt requires current information
- For real-time queries:
  1. Fetches relevant search results
  2. Incorporates results into model prompts
  3. Models explain how they used the information
  4. Evaluations consider the use of real-time data

### Round-Robin Evaluation
- GPT-4 evaluates Claude 3
- Claude 3 evaluates Gemini 1.5
- Gemini 1.5 evaluates GPT-4
- Each evaluation includes:
  - Detailed reasoning
  - Metric scores
  - Additional observations

### Data Management
- **CSV Structure**:
  - Timestamp
  - Prompt
  - Model
  - Evaluator
  - Response
  - All metrics
  - Reasoning
  - Notes
- **File Organization**:
  - Results stored in `results/` directory
  - Files named `ai_prompt_eval_YYYYMMDD_HHMMSS.csv`
  - Easy to track and compare evaluations

## Error Handling
- Graceful handling of API failures
- Fallback mechanisms for evaluation
- Detailed error logging
- User-friendly error messages

## Contributing
Feel free to submit issues and enhancement requests!

## License
[Your chosen license]

## Acknowledgments
- OpenAI for GPT-4
- Anthropic for Claude 3
- Google for Gemini 1.5
- Gradio for the UI framework 