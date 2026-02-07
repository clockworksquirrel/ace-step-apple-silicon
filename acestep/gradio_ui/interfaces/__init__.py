"""
Gradio UI Components Module
Contains all Gradio interface component definitions and layouts
"""
import gradio as gr
from acestep.gradio_ui.i18n import get_i18n, t
from acestep.gradio_ui.interfaces.dataset import create_dataset_section
from acestep.gradio_ui.interfaces.generation import create_generation_section
from acestep.gradio_ui.interfaces.result import create_results_section
from acestep.gradio_ui.interfaces.training import create_training_section
from acestep.gradio_ui.interfaces.dj_mode import create_dj_mode_section
from acestep.gradio_ui.events import setup_event_handlers, setup_training_event_handlers


def create_gradio_interface(dit_handler, llm_handler, dataset_handler, init_params=None, language='en') -> gr.Blocks:
    """
    Create Gradio interface
    
    Args:
        dit_handler: DiT handler instance
        llm_handler: LM handler instance
        dataset_handler: Dataset handler instance
        init_params: Dictionary containing initialization parameters and state.
                    If None, service will not be pre-initialized.
        language: UI language code ('en', 'zh', 'ja', default: 'en')
        
    Returns:
        Gradio Blocks instance
    """
    # Initialize i18n with selected language
    i18n = get_i18n(language)
    
    with gr.Blocks(
        title=t("app.title"),
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.cyan,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        ).set(
            body_background_fill="#0f0f14",
            body_background_fill_dark="#0f0f14",
            block_background_fill="#1a1a24",
            block_background_fill_dark="#1a1a24",
            block_border_width="1px",
            block_border_color="#2a2a3a",
            block_border_color_dark="#2a2a3a",
            block_label_background_fill="#1a1a24",
            block_label_background_fill_dark="#1a1a24",
            block_title_text_color="#e0e0e0",
            block_title_text_color_dark="#e0e0e0",
            body_text_color="#c8c8d0",
            body_text_color_dark="#c8c8d0",
            input_background_fill="#12121a",
            input_background_fill_dark="#12121a",
            input_border_color="#2a2a3a",
            input_border_color_dark="#2a2a3a",
            button_primary_background_fill="#0ea5e9",
            button_primary_background_fill_dark="#0ea5e9",
            button_primary_text_color="#ffffff",
            button_secondary_background_fill="#1e1e2e",
            button_secondary_background_fill_dark="#1e1e2e",
            button_secondary_text_color="#c8c8d0",
            shadow_drop="none",
            shadow_drop_lg="none",
        ),
        css="""
        /* ACE-Step Studio — Dark Production Theme */
        .gradio-container {
            max-width: 1400px !important;
            margin: auto;
        }
        .main-header {
            text-align: center;
            margin-bottom: 1.5rem;
            padding: 2rem 0 1rem;
            border-bottom: 1px solid #2a2a3a;
        }
        .main-header h1 {
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }
        .main-header p {
            color: #8888a0;
            font-size: 1.1rem;
        }
        .section-header {
            background: linear-gradient(135deg, #0ea5e9, #6366f1);
            color: white;
            padding: 10px 16px;
            border-radius: 8px;
            margin: 10px 0;
            font-weight: 600;
        }
        /* Tab styling */
        .tab-nav button {
            font-weight: 500 !important;
            border-radius: 8px 8px 0 0 !important;
        }
        .tab-nav button.selected {
            background: #1a1a24 !important;
            border-bottom: 2px solid #0ea5e9 !important;
        }
        /* Slider and input refinements */
        input[type="range"] {
            accent-color: #0ea5e9;
        }
        /* Audio player */
        audio {
            border-radius: 8px;
            width: 100%;
        }
        /* Accordion headers */
        .label-wrap {
            font-weight: 600 !important;
        }
        /* Buttons */
        button.primary {
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        button.primary:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3) !important;
        }
        button.secondary {
            border-radius: 8px !important;
        }
        /* LM hints alignment */
        .lm-hints-row { align-items: stretch; }
        .lm-hints-col { display: flex; }
        .lm-hints-col > div { flex: 1; display: flex; }
        .lm-hints-btn button { height: 100%; width: 100%; }
        .component-wrapper > .timestamps { transform: translateY(15px); }
        /* Footer */
        .footer-note {
            text-align: center;
            color: #555;
            font-size: 0.85rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #2a2a3a;
        }
        """,
    ) as demo:
        
        gr.HTML(f"""
        <div class="main-header">
            <h1>ACE-Step Studio</h1>
            <p>AI Music Generation — Apple Silicon Optimized</p>
        </div>
        """)
        
        # Dataset Explorer Section
        dataset_section = create_dataset_section(dataset_handler)
        
        # Generation Section (pass init_params and language to support pre-initialization)
        generation_section = create_generation_section(dit_handler, llm_handler, init_params=init_params, language=language)
        
        # Results Section
        results_section = create_results_section(dit_handler)
        
        # Training Section (LoRA training and dataset builder)
        # Pass init_params to support hiding in service mode
        training_section = create_training_section(dit_handler, llm_handler, init_params=init_params)
        
        # DJ Mode Section (experimental — LLM-powered setlist planning + generation)
        dj_section = create_dj_mode_section(dit_handler, llm_handler, init_params=init_params)
        
        # Connect event handlers
        setup_event_handlers(demo, dit_handler, llm_handler, dataset_handler, dataset_section, generation_section, results_section)
        
        # Connect training event handlers
        setup_training_event_handlers(demo, dit_handler, llm_handler, training_section)
    
    return demo
