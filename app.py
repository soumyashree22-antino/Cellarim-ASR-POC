import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import shutil

# Make the src/ package importable
ROOT = Path.cwd()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from asr_poc.config import load_config
from asr_poc import phylo, feature_table, ranking, report, embeddings, structure, structure_scoring, llm_scoring, io_utils

st.set_page_config(page_title="Cellarim ASR POC", layout="wide")

st.title("🧬 Cellarim ASR POC: Enzyme Engineering Pipeline")
st.markdown("Upload your sequences and run the end-to-end AI pipeline to engineer and rank ancestral candidates!")

# Sidebar configuration
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload FASTA file (e.g. lipases)", type=["fasta", "fa"])
skip_folding = st.sidebar.checkbox("Skip 3D Structure Folding (ESMFold)", value=True, help="Streamlit Cloud limits RAM to 1GB. ESMFold requires high memory. Check this to skip structural validation.")

def run_pipeline(file_content, cfg):
    # 1. Setup Data
    user_fasta = cfg.paths.curated_fasta
    user_fasta.parent.mkdir(parents=True, exist_ok=True)
    user_fasta.write_bytes(file_content)
    
    st.info("Pipeline started. This may take several minutes depending on dataset size.")
    
    # 2. Alignment
    with st.status("Step 1: Aligning Sequences (MAFFT)...", expanded=False) as status:
        msa_path = phylo.align(cfg, in_fasta=user_fasta)
        status.update(label="Step 1: Alignment Complete", state="complete")
        
    # 3. Phylogeny
    with st.status("Step 2: Building Phylogenetic Tree (IQ-TREE)...", expanded=False) as status:
        # Note: We temporarily force fast-mode via config to ensure it runs fast on Streamlit
        cfg.phylogeny.ultrafast_bootstrap = 0 
        cfg.phylogeny.iqtree_model = "LG"
        tree_path = phylo.build_tree(cfg, msa=msa_path)
        status.update(label="Step 2: Tree Built", state="complete")
        
    # 4. ASR
    with st.status("Step 3: Reconstructing Ancestors...", expanded=False) as status:
        state_file = phylo.reconstruct_ancestors(cfg, msa=msa_path)
        summary = phylo.build_candidate_pool(cfg, state_file=state_file)
        status.update(label=f"Step 3: ASR Complete ({summary['candidates']} candidates generated)", state="complete")
        
    # 5. Embeddings & Feature Extraction
    with st.status("Step 4: AI Feature Extraction (ESM-2)...", expanded=False) as status:
        signals = ranking.pre_fold_rank(cfg)
        top_k_ids = ranking.candidates_to_fold(signals, cfg)
        top_k = signals.loc[top_k_ids]
        
        # Save initial ranking
        top_k_csv = cfg.paths.reports_dir / "candidate_ranking.csv"
        top_k_csv.parent.mkdir(parents=True, exist_ok=True)
        top_k.to_csv(top_k_csv, index=True)
        final_candidates = top_k
        status.update(label="Step 4: AI Feature Extraction Complete", state="complete")

    # 6. Structure Prediction (Optional)
    if not skip_folding:
        with st.status("Step 5: 3D Structure Prediction (ESMFold)...", expanded=False) as status:
            struct_metrics = structure.analyze_candidates(cfg, ranking=top_k)
            final_candidates = ranking.final_rank(signals, struct_metrics, cfg)
            final_candidates.to_csv(top_k_csv, index=True)
            status.update(label="Step 5: 3D Structure Validated", state="complete")
    else:
        st.info("Skipping 3D structure prediction step.")
        
    # 7. LLM Scoring & Reporting
    with st.status("Step 6: Generating Scientific Report...", expanded=False) as status:
        try:
            report_path = report.write_report(cfg, final_candidates)
            status.update(label="Step 6: Report Generated", state="complete")
            return report_path, top_k_csv
        except Exception as e:
            import traceback
            st.error(f"Error during report generation:\n\n{traceback.format_exc()}")
            status.update(label="Step 6: Report Failed", state="error")
            return None, top_k_csv


# UI Logic
if uploaded_file is not None:
    if st.sidebar.button("🚀 Run AI Engineering Pipeline", use_container_width=True):
        # Clean previous runs
        cfg = load_config(ROOT / "config" / "target.yaml")
        
        # Clear phylogeny directory so IQ-TREE doesn't resume old runs
        if cfg.paths.phylogeny_dir.exists():
            shutil.rmtree(cfg.paths.phylogeny_dir)
        cfg.paths.phylogeny_dir.mkdir(parents=True, exist_ok=True)
            
        report_path, ranking_path = run_pipeline(uploaded_file.getvalue(), cfg)
        
        if ranking_path and ranking_path.exists():
            st.success("Pipeline execution finished successfully!")
            st.balloons()
else:
    st.info("👈 Please upload a FASTA file in the sidebar to begin.")


# Display Results
st.divider()

st.header("🏆 Ranked Candidates")
ranking_csv = Path("reports/candidate_ranking.csv")
if ranking_csv.exists():
    df = pd.read_csv(ranking_csv)
    st.dataframe(df, use_container_width=True)
else:
    st.write("No ranking data yet.")

st.divider()

st.header("📄 Scientific Report")
report_md = Path("reports/scientific_report.md")
if report_md.exists():
    st.markdown(report_md.read_text())
else:
    st.write("No scientific report generated yet.")
