import pandas as pd
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
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
    def __init__(self, question: str, answer: str, reason: str, transcript_id: str):
        self.question = question
        self.answer = answer
        self.reason = reason
        self.transcript_id = transcript_id


@dataclass
class NodeInfo:
    content: str
    node_type: str
    transcript_ids: Set[str]
    multiplicity: int


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
    
    def _get_or_create_node_id(self, content: str, node_type: str, transcript_id: str) -> int:
        """Get existing node ID or create new one for given content."""
        if content in self.content_to_node_id:
            node_id = self.content_to_node_id[content]
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
            multiplicity=1
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
                qa_pair.question, "question", qa_pair.transcript_id
            )
            answer_id = self._get_or_create_node_id(
                qa_pair.answer, "answer", qa_pair.transcript_id
            )
            reason_id = self._get_or_create_node_id(
                qa_pair.reason, "reason", qa_pair.transcript_id
            )
            
            # Add edges within QA pair
            self._add_edge(question_id, answer_id, "question_to_answer")
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
                "node_multiplicity": node_info.multiplicity
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
                transcript_id=row['transcript_id']
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
            transcript_id=row['transcript_id']
        )
    
    # Add edges
    for _, row in qa_edges.iterrows():
        G.add_edge(
            row['source_node_id'],
            row['target_node_id'],
            edge_type=row['edge_type'],
            multiplicity=row['edge_multiplicity']
        )
    
    # Calculate layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
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
    gephi_nodes = gephi_nodes[['Id', 'Label', 'content', 'node_type', 'transcript_id', 'node_multiplicity']]
    
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
    nodes_path = os.path.join(output_dir, "qa_nodes_gephi.csv")
    edges_path = os.path.join(output_dir, "qa_edges_gephi.csv")
    
    gephi_nodes.to_csv(nodes_path, index=False, encoding='utf-8')
    gephi_edges.to_csv(edges_path, index=False, encoding='utf-8')
    
    # Create import instructions
    instructions_path = os.path.join(output_dir, "gephi_import_instructions.md")
    instructions = f"""
        # Gephi Import Instructions

        ## Files Created:
        - **qa_nodes_gephi.csv**: Node data for import
        - **qa_edges_gephi.csv**: Edge data for import

        ## How to Import into Gephi:

        ### Step 1: Import Nodes
        1. Open Gephi
        2. Go to **File → Import Spreadsheet**
        3. Select `qa_nodes_gephi.csv`
        4. Choose **"Nodes table"** as import type
        5. Make sure **"Id"** column is set as node identifier
        6. Click **"Next"** and then **"Finish"**

        ### Step 2: Import Edges
        1. Go to **File → Import Spreadsheet** again
        2. Select `qa_edges_gephi.csv`
        3. Choose **"Edges table"** as import type
        4. Make sure **"Source"** and **"Target"** columns are properly mapped
        5. Set **"Weight"** column as edge weight if desired
        6. Click **"Next"** and then **"Finish"**

        ## Recommended Gephi Settings:

        ### Layout Algorithms:
        - **ForceAtlas 2**: Good for general network visualization
        - **Fruchterman Reingold**: Classic force-directed layout
        - **Yifan Hu**: Good for larger networks

        ### Node Styling:
        - **Size**: Based on `node_multiplicity` column
        - **Color**: Based on `node_type` column (question/answer/reason)

        ### Edge Styling:
        - **Thickness**: Based on `Weight` (edge multiplicity)
        - **Color**: Based on `edge_type` column

        ### Useful Filters:
        - Filter by `node_type` to show only questions, answers, or reasons
        - Filter by `edge_type` to show specific relationship types
        - Filter by `transcript_id` to focus on specific transcripts

        ## Network Statistics:
        - Total Nodes: {len(gephi_nodes)}
        - Total Edges: {len(gephi_edges)}
        - Node Types: {gephi_nodes['node_type'].value_counts().to_dict()}
        - Edge Types: {gephi_edges['edge_type'].value_counts().to_dict()}
        """
    
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    return {
        "nodes_csv_path": nodes_path,
        "edges_csv_path": edges_path,
        "instructions_path": instructions_path,
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
        "summary_artifact_id": summary_artifact_id
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


# # Example usage and testing
# def create_sample_data() -> List[QAPair]:
#     """Create sample QA pairs for testing."""
#     return [
#         QAPair("What is AI?", "Artificial Intelligence", "It's a broad field", "1"),
#         QAPair("What is ML?", "Machine Learning", "It's a subset of AI", "1"),
#         QAPair("What is AI?", "Artificial Intelligence", "It enables machines to think", "2"),
#         QAPair("How does ML work?", "Through algorithms", "It learns from data", "2"),
#         QAPair("What is deep learning?", "Neural networks with many layers", "It's a subset of AI", "3"),
#     ]


if __name__ == "__main__":
    # Example usage
    # sample_qa_pairs = create_sample_data()
    processed_qa_pairs = import_qa_pairs_from_csv('qa_pairs.csv')
    result = qa_network_pipeline(processed_qa_pairs, export_gephi=True, output_dir="my_gephi_files")


    
    print("Nodes DataFrame:")
    print(result["qa_nodes"])
    print("\nEdges DataFrame:")
    print(result["qa_edges"])
    print(f"\nArtifact IDs: {result['artifact_ids']}")
    
    if result["gephi_export"]:
        print(f"\nGephi Export Info:")
        print(f"Nodes CSV: {result['gephi_export']['nodes_csv_path']}")
        print(f"Edges CSV: {result['gephi_export']['edges_csv_path']}")
        print(f"Instructions: {result['gephi_export']['instructions_path']}")
        print(f"Exported {result['gephi_export']['node_count']} nodes and {result['gephi_export']['edge_count']} edges")