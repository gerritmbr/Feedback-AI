import pandas as pd
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import hashlib
import json
from prefect import task, flow
from prefect.artifacts import create_table_artifact, create_markdown_artifact
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import csv


class QAPair:
    def __init__(self, question: str, answer: str, reason: str, transcript_id: str, content_category: str = ""):
        self.question = question
        self.answer = answer
        self.reason = reason
        self.transcript_id = transcript_id
        self. content_category = content_category


@dataclass
class NodeInfo:
    content: str
    node_type: str
    transcript_ids: Set[str]
    multiplicity: int
    content_category: str


@dataclass
class EdgeInfo:
    source_node_id: int
    target_node_id: int
    edge_type: str
    multiplicity: int


class QANetworkBuilder:
    """Utility class to build network graph structures from QA pairs."""
    
    def __init__(self):
        self.content_to_node_id: Dict[str, int] = {}
        self.node_info: Dict[int, NodeInfo] = {}
        self.edges: Dict[Tuple[int, int, str], int] = {}  # (source, target, type) -> multiplicity
        self.next_node_id = 1

    def _get_or_create_node_id(self, content: str, node_type: str, transcript_id: str, content_category: str) -> int:
        """Get existing node ID or create new one for given content."""
        if content in self.content_to_node_id:
            node_id = self.content_to_node_id[content]
            # Only increment multiplicity if this transcript_id is new for the given content
            if transcript_id not in self.node_info[node_id].transcript_ids:
                self.node_info[node_id].transcript_ids.add(transcript_id)
                self.node_info[node_id].multiplicity += 1
            return node_id
    
        node_id = self.next_node_id
        self.next_node_id += 1
        self.content_to_node_id[content] = node_id
        self.node_info[node_id] = NodeInfo(
            content=content,
            node_type=node_type,
            transcript_ids={transcript_id},
            multiplicity=1,
            content_category= content_category
        )
        return node_id
    
    def _add_edge(self, source_id: int, target_id: int, edge_type: str):
        """Add or increment edge between two nodes."""
        edge_key = (source_id, target_id, edge_type)
        self.edges[edge_key] = self.edges.get(edge_key, 0) + 1
    
    def build_network(self, qa_pairs: List[QAPair]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build network graph from QA pairs."""
        
        # Step 1: Create nodes and basic edges (question -> answer -> reason)
        for qa_pair in qa_pairs:
            question_id = self._get_or_create_node_id(
                qa_pair.question, "question", qa_pair.transcript_id, qa_pair.content_category
            )
            answer_id = self._get_or_create_node_id(
                qa_pair.answer, "answer", qa_pair.transcript_id, qa_pair.content_category
            )

            self._add_edge(question_id, answer_id, "question_to_answer")

            # Only create reason node and edge if reason is not "n/a"
            if qa_pair.reason and qa_pair.reason.lower() != "n/a":

                reason_id = self._get_or_create_node_id(
                    qa_pair.reason, "reason", qa_pair.transcript_id, qa_pair.content_category
                )
            
                self._add_edge(answer_id, reason_id, "answer_to_reason")
        
        # Step 2: Create edges between answers/reasons with same transcript_id
        self._create_same_transcript_edges()
        
        # Step 3: Convert to DataFrames
        return self._to_dataframes()
    
    def _create_same_transcript_edges(self):
        """Create edges between nodes of same type sharing transcript IDs."""
        
        # Group nodes by type and transcript_id
        transcript_answers = defaultdict(list)
        transcript_reasons = defaultdict(list)
        
        for node_id, node_info in self.node_info.items():
            for transcript_id in node_info.transcript_ids:
                if node_info.node_type == "answer":
                    transcript_answers[transcript_id].append(node_id)
                elif node_info.node_type == "reason":
                    transcript_reasons[transcript_id].append(node_id)
        
        # Create edges between answers in same transcript
        for transcript_id, answer_ids in transcript_answers.items():
            for i in range(len(answer_ids)):
                for j in range(i + 1, len(answer_ids)):
                    self._add_edge(answer_ids[i], answer_ids[j], "same_transcript_answer")
        
        # Create edges between reasons in same transcript
        for transcript_id, reason_ids in transcript_reasons.items():
            for i in range(len(reason_ids)):
                for j in range(i + 1, len(reason_ids)):
                    self._add_edge(reason_ids[i], reason_ids[j], "same_transcript_reason")
    
    def _to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert internal structures to DataFrames."""
        
        # Create nodes DataFrame
        nodes_data = []
        for node_id, node_info in self.node_info.items():
            # For nodes appearing in multiple transcripts, join transcript IDs
            transcript_id_str = "|".join(sorted(node_info.transcript_ids))
            nodes_data.append({
                "node_id": node_id,
                "content": node_info.content,
                "node_type": node_info.node_type,
                "transcript_id": transcript_id_str,
                "node_multiplicity": node_info.multiplicity,
                "content_category": node_info.content_category
            })
        
        qa_nodes = pd.DataFrame(nodes_data).sort_values("node_id")
        
        # Create edges DataFrame
        edges_data = []
        for (source_id, target_id, edge_type), multiplicity in self.edges.items():
            edges_data.append({
                "source_node_id": source_id,
                "target_node_id": target_id,
                "edge_type": edge_type,
                "edge_multiplicity": multiplicity
            })
        
        qa_edges = pd.DataFrame(edges_data)
        
        return qa_nodes, qa_edges

@task(name="import-qa-data")
def import_qa_pairs_from_csv(filename: str) -> list[QAPair]:
    qa_pairs = []
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            qa_pair = QAPair(
                question=row['question'],
                answer=row['answer'],
                reason=row['reason'],
                transcript_id=row['transcript_id'],
                content_category=row['content_category']
            )
            qa_pairs.append(qa_pair)
    return qa_pairs


@task(name="build-qa-network")
def build_qa_network_task(qa_pairs: List[QAPair]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prefect task to build QA network graph."""
    builder = QANetworkBuilder()
    qa_nodes, qa_edges = builder.build_network(qa_pairs)
    return qa_nodes, qa_edges


@task(name="create_network_visualization")
def create_network_visualization_task(qa_nodes: pd.DataFrame, qa_edges: pd.DataFrame) -> go.Figure:
    """Create interactive network visualization using Plotly and NetworkX."""
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for _, row in qa_nodes.iterrows():
        G.add_node(
            row['node_id'],
            content=row['content'][:50] + "..." if len(row['content']) > 50 else row['content'],
            node_type=row['node_type'],
            multiplicity=row['node_multiplicity'],
            transcript_id=row['transcript_id'],
            content_category=row['content_category']
        )
    
    # Add edges
    for _, row in qa_edges.iterrows():
        G.add_edge(
            row['source_node_id'],
            row['target_node_id'],
            edge_type=row['edge_type'],
            multiplicity=row['edge_multiplicity']
        )
    
    # TBD - make algorithm selection an input parameter
    # Calculate layout 
    # Fruchterman Rheingold
    # pos = nx.spring_layout(G, k=3, iterations=50)

    # Force Atlas 2
    pos = nx.forceatlas2_layout(G)

    # Create edge traces
    edge_traces = []
    edge_colors = {
        'question_to_answer': 'blue',
        'answer_to_reason': 'green',
        'same_transcript_answer': 'orange',
        'same_transcript_reason': 'purple'
    }
    
    for edge_type, color in edge_colors.items():
        edge_x, edge_y = [], []
        edge_info = []
        
        for edge in G.edges(data=True):
            if edge[2]['edge_type'] == edge_type:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_info.append(f"Multiplicity: {edge[2]['multiplicity']}")
        
        if edge_x:  # Only add trace if there are edges of this type
            edge_traces.append(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color=color),
                hoverinfo='none',
                mode='lines',
                name=f'{edge_type.replace("_", " ").title()} Edges',
                showlegend=True
            ))
    
    # Create node trace
    node_x = []
    node_y = []
    node_info = []
    node_colors = []
    node_sizes = []
    
    color_map = {'question': 'lightblue', 'answer': 'lightgreen', 'reason': 'lightcoral'}
    
    for node_id in G.nodes():
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        
        node_data = G.nodes[node_id]
        node_info.append(
            f"ID: {node_id}<br>"
            f"Type: {node_data['node_type']}<br>"
            f"Multiplicity: {node_data['multiplicity']}<br>"
            f"Content: {node_data['content']}<br>"
            f"Transcript: {node_data['transcript_id']}"
            f"Category: {node_data['content_category']}"
        )
        node_colors.append(color_map[node_data['node_type']])
        node_sizes.append(max(10, min(30, node_data['multiplicity'] * 5)))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_info,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=2, color='black')
        ),
        name='Nodes',
        showlegend=True
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        # title='QA Network Graph',
                        # titlefont_size=16,
                        title=dict(
                            text='QA Network Graph',
                            font=dict(size=16)
                        ),
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Node size indicates multiplicity. Colors: Blue=Question, Green=Answer, Red=Reason",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    return fig


@task(name="export-for-gephi")
def export_for_gephi_task(
    qa_nodes: pd.DataFrame, 
    qa_edges: pd.DataFrame,
    output_dir: str = "gephi_export"
) -> Dict[str, str]:
    """Export network data as CSV files optimized for Gephi import."""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare nodes for Gephi (Gephi expects 'Id' and 'Label' columns)
    gephi_nodes = qa_nodes.copy()
    gephi_nodes = gephi_nodes.rename(columns={'node_id': 'Id'})
    gephi_nodes['Label'] = gephi_nodes['content'].str[:50] + "..."  # Truncate for readability
    
    # Reorder columns for Gephi (Id and Label should be first)
    gephi_nodes = gephi_nodes[['Id', 'Label', 'content', 'node_type', 'transcript_id', 'node_multiplicity', 'content_category']]
    
    # Prepare edges for Gephi (Gephi expects 'Source' and 'Target' columns)
    gephi_edges = qa_edges.copy()
    gephi_edges = gephi_edges.rename(columns={
        'source_node_id': 'Source',
        'target_node_id': 'Target'
    })
    
    # Add weight column for Gephi (based on edge multiplicity)
    gephi_edges['Weight'] = gephi_edges['edge_multiplicity']
    
    # Reorder columns for Gephi
    gephi_edges = gephi_edges[['Source', 'Target', 'Weight', 'edge_type', 'edge_multiplicity']]
    
    # Export to CSV
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    nodes_path = os.path.join(output_dir, f"qa_nodes_gephi_{current_time}.csv")
    edges_path = os.path.join(output_dir, f"qa_edges_gephi_{current_time}.csv")
    
    gephi_nodes.to_csv(nodes_path, index=False, encoding='utf-8')
    gephi_edges.to_csv(edges_path, index=False, encoding='utf-8')
    
    
    return {
        "nodes_csv_path": nodes_path,
        "edges_csv_path": edges_path,
        "node_count": len(gephi_nodes),
        "edge_count": len(gephi_edges)
    }


@task(name="save-qa-network-artifacts")
def save_qa_network_artifacts_task(
    qa_nodes: pd.DataFrame, 
    qa_edges: pd.DataFrame, 
    visualization: go.Figure
) -> Dict[str, str]:
    """Save QA network data and visualization as Prefect artifacts."""
    import tempfile
    import os
    from prefect.artifacts import create_link_artifact
    # Save nodes table
    nodes_artifact_id = create_table_artifact(
        key="qa-nodes-table",
        table=qa_nodes.to_dict('records'),
        description="QA Network Nodes - Contains all unique questions, answers, and reasons with their metadata"
    )
    
    # Save edges table
    edges_artifact_id = create_table_artifact(
        key="qa-edges-table",
        table=qa_edges.to_dict('records'),
        description="QA Network Edges - Contains all connections between nodes with relationship types"
    )
    
    # Save visualization as HTML artifact
    visualization_artifact_id = None
    try:
        # Create temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
            # Generate HTML content
            html_content = visualization.to_html(
                include_plotlyjs=True,  # Include Plotly.js for standalone file
                div_id="qa-network-graph",
                config={'displayModeBar': True, 'responsive': True}
            )
            temp_file.write(html_content)
            temp_html_path = temp_file.name
        
        # Read the HTML content
        with open(temp_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Create HTML artifact using markdown (with HTML content)
        visualization_artifact_id = create_markdown_artifact(
            key="qa-network-visualization", 
            markdown=f"""# QA Network Interactive Visualization

        <div style="width: 100%; height: 600px; border: 1px solid #ddd; border-radius: 4px; overflow: hidden;">

        {html_content}

        </div>

        ## How to Use This Visualization:
        - **Zoom**: Use mouse wheel or zoom controls
        - **Pan**: Click and drag to move around
        - **Hover**: Hover over nodes and edges for details
        - **Legend**: Click legend items to show/hide elements
        - **Reset**: Double-click to reset zoom

        ## Node Information:
        - **Blue nodes**: Questions
        - **Green nodes**: Answers  
        - **Red nodes**: Reasons
        - **Node size**: Proportional to multiplicity (how often it appears)

        ## Edge Information:
        - **Blue edges**: Question → Answer connections
        - **Green edges**: Answer → Reason connections
        - **Orange edges**: Same transcript answer connections
        - **Purple edges**: Same transcript reason connections
        """,
            description="Interactive QA Network Graph - Hover over nodes for details, use controls to zoom/pan"
        )
        
        # Clean up temporary file
        os.unlink(temp_html_path)
        
    except Exception as e:
        print(f"Warning: Could not create visualization artifact: {e}")
    
    # Also create a downloadable HTML file artifact
    html_file_artifact_id = None
    try:
        # Save as a file that can be downloaded
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"output_data/qa_network_interactive_{current_time}.html"
        visualization.write_html(html_filename)
        
        # Create link artifact pointing to the file
        html_file_artifact_id = create_link_artifact(
            key="qa-network-html-download",
            link=f"./{html_filename}",
            description=f"Downloadable HTML file: {html_filename} - Open in browser for full interactive experience"
        )
        
    except Exception as e:
        print(f"Warning: Could not create HTML file artifact: {e}")
    

    # Create summary statistics
    summary_stats = f"""
        # QA Network Analysis Summary

        ## Network Statistics
        - **Total Nodes**: {len(qa_nodes)}
        - **Total Edges**: {len(qa_edges)}
        - **Node Types**: {qa_nodes['node_type'].value_counts().to_dict()}
        - **Edge Types**: {qa_edges['edge_type'].value_counts().to_dict()}

        ## Node Multiplicity Analysis
        - **Average Node Multiplicity**: {qa_nodes['node_multiplicity'].mean():.2f}
        - **Max Node Multiplicity**: {qa_nodes['node_multiplicity'].max()}
        - **Nodes with Multiplicity > 1**: {(qa_nodes['node_multiplicity'] > 1).sum()}

        ## Edge Multiplicity Analysis
        - **Average Edge Multiplicity**: {qa_edges['edge_multiplicity'].mean():.2f}
        - **Max Edge Multiplicity**: {qa_edges['edge_multiplicity'].max()}
        - **Edges with Multiplicity > 1**: {(qa_edges['edge_multiplicity'] > 1).sum()}

        ## Transcript Coverage
        - **Unique Transcripts in Nodes**: {qa_nodes['transcript_id'].str.split('|').explode().nunique()}


        ## Visualization Files Created:
        - Interactive HTML visualization (embedded above)
        - Downloadable HTML file: `qa_network_interactive_{current_time}.html`
        - CSV files for Gephi import (if enabled)
            """
    
    summary_artifact_id = create_markdown_artifact(
        key="qa-network-summary",
        markdown=summary_stats,
        description="Statistical summary of the QA network graph"
    )
    
    # Note: Plotly figures can't be directly saved as Prefect artifacts
    # In a real implementation, you might save as HTML or PNG
    
    return {
        "nodes_artifact_id": nodes_artifact_id,
        "edges_artifact_id": edges_artifact_id,
        "summary_artifact_id": summary_artifact_id,
        "visualization_artifact_id": visualization_artifact_id,
        "html_file_artifact_id": html_file_artifact_id
    }


@flow(name="qa-network-pipeline")
def qa_network_pipeline(qa_pairs: List[QAPair], export_gephi: bool = True, output_dir: str = "gephi_export") -> Dict:
    """Complete Prefect flow for QA network analysis."""
    
    # Build network
    qa_nodes, qa_edges = build_qa_network_task(qa_pairs)
    
    # Create visualization
    visualization = create_network_visualization_task(qa_nodes, qa_edges)


    # Save artifacts
    artifact_ids = save_qa_network_artifacts_task(qa_nodes, qa_edges, visualization)


    
    # Export for Gephi if requested
    gephi_export_info = None
    if export_gephi:
        gephi_export_info = export_for_gephi_task(qa_nodes, qa_edges, output_dir)
    
    return {
        "qa_nodes": qa_nodes,
        "qa_edges": qa_edges,
        "visualization": visualization,
        "artifact_ids": artifact_ids,
        "gephi_export": gephi_export_info
    }


if __name__ == "__main__":
    # Example usage
    # sample_qa_pairs = create_sample_data()
    # processed_qa_pairs = import_qa_pairs_from_csv('qa_pairs.csv')
    processed_qa_pairs = import_qa_pairs_from_csv('output_data/qa_pairs_cleaned_and_flattened_20250624_092926.csv') #TBD turn this into an input variable to the script.
    result = qa_network_pipeline(processed_qa_pairs, export_gephi=True, output_dir="output_data/my_gephi_files")

    # All artifacts are automatically created
    print("Created artifacts:")
    for key, artifact_id in result["artifact_ids"].items():
        print(f"  {key}: {artifact_id}")

    
    print("Nodes DataFrame:")
    print(result["qa_nodes"])
    print("\nEdges DataFrame:")
    print(result["qa_edges"])
    print(f"\nArtifact IDs: {result['artifact_ids']}")
    
    if result["gephi_export"]:
        print(f"\nGephi Export Info:")
        print(f"Nodes CSV: {result['gephi_export']['nodes_csv_path']}")
        print(f"Edges CSV: {result['gephi_export']['edges_csv_path']}")
        print(f"Exported {result['gephi_export']['node_count']} nodes and {result['gephi_export']['edge_count']} edges")