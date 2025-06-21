# gradio_full_llm_eval.py – Final Updated Version with ATS Scoring and Visualized UI
import gradio as gr
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import zipfile
import json
from datetime import datetime
from dotenv import load_dotenv

from response_generator import generate_all_responses_with_reasoning
from round_robin_evaluator import comprehensive_round_robin_evaluation
from realtime_detector import is_realtime_prompt
from search_fallback import get_google_snippets

load_dotenv()
pio.kaleido.scope.default_format = "png"

metrics = ['helpfulness', 'correctness', 'coherence', 'tone_score',
           'accuracy', 'relevance', 'completeness', 'clarity']

def extract_text_from_resume(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".pdf":
        import fitz
        with fitz.open(file.name) as doc:
            return "\n".join(page.get_text() for page in doc)
    elif ext == ".docx":
        import docx
        doc = docx.Document(file.name)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".txt":
        return file.read().decode('utf-8')
    return ""

def ats_score_advanced(response, resume, jd):
    prompt = f"""
You are a professional ATS scoring engine. Compare the generated response to the candidate's resume and job description using:
1. Keyword Matching
2. Section Weighting
3. Semantic Similarity
4. Recency/Frequency
5. Penalty Detection
6. Aggregation

Resume:
{resume}

Job Description:
{jd}

Response:
{response}

Return JSON:
{{"ats_score": <0-100>, "strengths": ["..."], "gaps": ["..."], "suggestions": ["..."]}}
"""
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        res = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return json.loads(res.choices[0].message.content.strip())
    except:
        return {"ats_score": 50, "strengths": [], "gaps": [], "suggestions": ["Check formatting."]}

def create_visualizations(df, results_dir):
    image_files = []
    summary = df.groupby('target_model')[metrics].mean().reset_index()

    heatmap = px.imshow(summary[metrics].values, x=metrics, y=summary['target_model'],
                        labels=dict(x="Metric", y="Model", color="Score"),
                        title="Heatmap: Metrics Across Models", color_continuous_scale='Viridis')
    heatmap_path = os.path.join(results_dir, "heatmap.png")
    heatmap.write_image(heatmap_path)
    image_files.append(heatmap_path)

    radar = go.Figure()
    for _, row in summary.iterrows():
        radar.add_trace(go.Scatterpolar(r=list(row[metrics]), theta=metrics, fill='toself', name=row['target_model']))
    radar.update_layout(title="Radar Chart: Model Score Profiles", polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    radar_path = os.path.join(results_dir, "radar.png")
    radar.write_image(radar_path)
    image_files.append(radar_path)

    bar = px.bar(summary.melt(id_vars='target_model'), x='variable', y='value', color='target_model', barmode='group',
                 title="Bar Chart: Metric Comparison")
    bar_path = os.path.join(results_dir, "barchart.png")
    bar.write_image(bar_path)
    image_files.append(bar_path)

    return (heatmap, radar, bar), image_files

def format_ats_feedback(score, strengths, gaps, suggestions):
    color = "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"
    return f"""
### ATS Match Score: ~{score}% {color}

#### **Strengths / High Matches:**
{chr(10).join([f"* {s}" for s in strengths]) if strengths else "* None found."}

#### **Partial or Missing:**
{chr(10).join([f"* {g}" for g in gaps]) if gaps else "* None mentioned."}

#### **How to Improve ATS Score:**
{chr(10).join([f"1. {s}" for s in suggestions]) if suggestions else "1. Add missing skills."}
"""

def process_prompt(prompt, enable_realtime, enable_eval, enable_analysis, user_file, model_selection):
    selected_models = [m for m, enabled in zip(["GPT-4", "Claude 3", "Gemini 1.5"], model_selection) if enabled]
    resume_text = ""
    batch_mode = user_file and user_file.name.endswith(".csv")
    resume_mode = user_file and user_file.name.lower().endswith(('.pdf', '.docx', '.txt'))

    prompts = [prompt]
    ats_summary_texts = []
    search_results = ""

    if batch_mode:
        df_batch = pd.read_csv(user_file.name)
        prompts = df_batch['prompt'].dropna().tolist()
    elif resume_mode:
        resume_text = extract_text_from_resume(user_file)

    all_rows, all_charts = [], []
    zip_path, ats_table_markdown = None, ""

    for prompt_text in prompts:
        search_results = get_google_snippets(prompt_text) if enable_realtime and is_realtime_prompt(prompt_text) else ""
        final_prompt = f"{prompt_text}\n\nRecent info: {search_results}" if search_results else prompt_text
        responses = generate_all_responses_with_reasoning(final_prompt, selected_models)

        ats_rows = []
        for model in responses:
            model_resp = responses[model]['response']
            if resume_text:
                ats_result = ats_score_advanced(model_resp, resume_text, prompt_text)
                feedback = format_ats_feedback(ats_result['ats_score'], ats_result.get('strengths', []), ats_result.get('gaps', []), ats_result.get('suggestions', []))
                responses[model]['ats_embed'] = f"###  Response\n\n{model_resp}\n\n---\n\n###  ATS Evaluation\n\n{feedback}"
                ats_rows.append(f"| {model} | {ats_result['ats_score']} | {', '.join(ats_result.get('strengths', []))} | {', '.join(ats_result.get('suggestions', []))} |")
            else:
                responses[model]['ats_embed'] = f"###  Response\n\n{model_resp}\n\n---\n\n**Explainability:**\n{responses[model]['reasoning']}"
        if ats_rows:
            ats_table_markdown = "| Model | Score | Strengths | Suggestions |\n|-------|-------|-----------|-------------|\n" + "\n".join(ats_rows)

        if enable_eval:
            compact = {k: v['response'] for k, v in responses.items()}
            eval_result = comprehensive_round_robin_evaluation(compact, final_prompt)
            for model, data in eval_result.items():
                for evaluator, scores in data['evaluations'].items():
                    row = {
                        'prompt': prompt_text,
                        'target_model': model,
                        'evaluator': evaluator,
                        'response': responses[model]['response'],
                        'explainability': responses[model]['reasoning']
                    }
                    row.update({k: scores.get(k, 0.5) for k in metrics})
                    row.update({f"avg_{k}": data['average_scores'].get(k, 0.5) for k in metrics})
                    all_rows.append(row)

    df_all = pd.DataFrame(all_rows)
    if not df_all.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/batch_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "evaluation.csv")
        df_all.to_csv(csv_path, index=False)
        (heatmap, radar, bar), chart_paths = create_visualizations(df_all, results_dir)
        all_charts = [heatmap, radar, bar]
        zip_path = os.path.join(results_dir, "bundle.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(csv_path, arcname="evaluation.csv")
            for chart in chart_paths:
                zipf.write(chart, arcname=os.path.basename(chart))
        if batch_mode:
            df_batch['ATS Summary'] = ats_summary_texts
            df_batch.to_csv(os.path.join(results_dir, "batch_prompts_output.csv"), index=False)
            zipf.write(os.path.join(results_dir, "batch_prompts_output.csv"), arcname="batch_prompts_output.csv")

    return tuple(
        responses[model].get('ats_embed', responses[model]['response']) for model in ["GPT-4", "Claude 3", "Gemini 1.5"]
    ) + (
        search_results or "N/A",
        *all_charts,
        df_all[['target_model', 'evaluator'] + metrics] if not df_all.empty else pd.DataFrame(),
        ats_table_markdown,
        zip_path
    )

def download_results(path):
    return path if path and os.path.exists(path) else None

def create_interface():
    with gr.Blocks(title="LLM Comparison Hub") as demo:
        gr.Markdown("""
# LLM Comparison Hub
This app compares LLM responses using round-robin evaluations, with real-time query detection and comprehensive analysis.

**How to use:**
- Enter a prompt (JD or query)
- Upload a resume (PDF/DOCX/TXT) or a CSV with prompts
- Select models
- Click evaluate

**Features:**
- Real-time web search fallback
- Resume vs JD ATS scoring (optional)
- Batch CSV prompt evaluation
- Visualizations (Heatmap, Radar, Bar)
- ZIP export of all results
""")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Enter Prompt", lines=4)
                user_file = gr.File(label="Upload Resume or CSV", file_types=[".pdf", ".docx", ".txt", ".csv"])
                model_selector = gr.CheckboxGroup(label="Select Models", choices=["GPT-4", "Claude 3", "Gemini 1.5"], value=["GPT-4", "Claude 3", "Gemini 1.5"])
                enable_realtime = gr.Checkbox(label="Enable real-time detection", value=True)
                enable_eval = gr.Checkbox(label="Enable evaluation", value=True)
                enable_analysis = gr.Checkbox(label="Enable analysis", value=True)
                submit = gr.Button("Run Evaluation")

            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("GPT-4"): gpt_out = gr.Markdown()
                    with gr.Tab("Claude 3"): claude_out = gr.Markdown()
                    with gr.Tab("Gemini 1.5"): gemini_out = gr.Markdown()
                    with gr.Tab("Evaluation Table"): df_out = gr.Dataframe()
                    with gr.Tab("ATS Evaluation"): ats_summary = gr.Markdown()
                    with gr.Tab("Search Results"): search_out = gr.Markdown()
                    with gr.Tab("Visualizations"):
                        heatmap_plot = gr.Plot()
                        radar_plot = gr.Plot()
                        bar_plot = gr.Plot()
                export_btn = gr.Button("Download ZIP Bundle")
                zip_output = gr.File(file_types=[".zip"], interactive=False, visible=True)

        submit.click(
            fn=process_prompt,
            inputs=[prompt, enable_realtime, enable_eval, enable_analysis, user_file, model_selector],
            outputs=[gpt_out, claude_out, gemini_out, search_out, heatmap_plot, radar_plot, bar_plot, df_out, ats_summary, zip_output]
        )
        export_btn.click(download_results, inputs=[zip_output], outputs=[zip_output])

    return demo

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
