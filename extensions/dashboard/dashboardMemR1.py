#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-R1 Dashboard
Advanced dashboard to visualize semantic graph memory, provenance chains and agent trace replay.
Integrates with graph_env.py and memory_r1_modular.py for comprehensive analysis.
"""

import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd

# Dashboard imports
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_cytoscape as cyto
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import Memory-R1 components
try:
    from memory_r1_modular import MemoryR1Enhanced, GraphTriple, GraphFragment, TraceEntry
    from graph_env import GraphMemoryEnv, PPOGraphTrainer
    MEMORY_R1_AVAILABLE = True
except ImportError:
    MEMORY_R1_AVAILABLE = False
    print("⚠️ Memory-R1 system not available for dashboard")

class MemoryR1Dashboard:
    """Advanced dashboard for Memory-R1 system visualization and analysis"""
    
    def __init__(self, memory_system: Optional[Any] = None, graph_env: Optional[Any] = None):
        self.memory_system = memory_system
        self.graph_env = graph_env
        
        # Dashboard configuration
        self.app = dash.Dash(__name__)
        self.app.title = "Memory-R1 Dashboard"
        
        # Data storage
        self.trace_data = []
        self.graph_data = {}
        self.provenance_data = {}
        self.validation_results = {}
        
        # Initialize dashboard
        self._setup_layout()
        self._setup_callbacks()
        
        print("📊 Memory-R1 Dashboard initialized")
    
    def _setup_layout(self):
        """Setup dashboard layout with multiple tabs"""
        
        self.app.layout = html.Div([
            html.Div([
                html.H1("Memory-R1 Advanced Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.P("Semantic Graph Memory • Provenance Chains • Agent Trace Replay",
                      style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),
            
            dcc.Tabs(id="main-tabs", value='graph-tab', children=[
                dcc.Tab(label='Semantic Graph', value='graph-tab'),
                dcc.Tab(label='Provenance Explorer', value='provenance-tab'),
                dcc.Tab(label='Trace Replay', value='trace-tab'),
                dcc.Tab(label='Validation & CI', value='validation-tab'),
                dcc.Tab(label='RL Training', value='rl-tab'),
                dcc.Tab(label='System Status', value='status-tab')
            ]),
            
            html.Div(id='tab-content', style={'padding': '20px'})
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('main-tabs', 'value')
        )
        def render_tab_content(active_tab):
            if active_tab == 'graph-tab':
                return self._render_graph_tab()
            elif active_tab == 'provenance-tab':
                return self._render_provenance_tab()
            elif active_tab == 'trace-tab':
                return self._render_trace_tab()
            elif active_tab == 'validation-tab':
                return self._render_validation_tab()
            elif active_tab == 'rl-tab':
                return self._render_rl_tab()
            elif active_tab == 'status-tab':
                return self._render_status_tab()
            else:
                return html.Div("Select a tab to view content")
        
        # Graph interaction callbacks
        @self.app.callback(
            Output('graph-info', 'children'),
            Input('semantic-graph', 'selectedNodeData')
        )
        def display_node_info(selected_nodes):
            if not selected_nodes:
                return "Select a node to view details"
            
            node = selected_nodes[0]
            return html.Div([
                html.H4(f"Node: {node.get('label', 'Unknown')}"),
                html.P(f"ID: {node.get('id', 'N/A')}"),
                html.P(f"Type: {node.get('node_type', 'entity')}"),
                html.P(f"Confidence: {node.get('confidence', 0.0):.3f}"),
                html.P(f"First Seen: {node.get('first_seen', 'Unknown')}")
            ])
        
        # Trace replay callbacks
        @self.app.callback(
            Output('trace-details', 'children'),
            Input('trace-slider', 'value')
        )
        def update_trace_details(trace_index):
            if not self.trace_data or trace_index >= len(self.trace_data):
                return "No trace data available"
            
            trace = self.trace_data[trace_index]
            return html.Div([
                html.H4(f"Trace {trace_index + 1}"),
                html.Pre(json.dumps(trace, indent=2, default=str),
                        style={'backgroundColor': '#f8f9fa', 'padding': '10px'})
            ])
        
        # Validation refresh callback
        @self.app.callback(
            Output('validation-results', 'children'),
            Input('refresh-validation', 'n_clicks')
        )
        def refresh_validation(n_clicks):
            if n_clicks and self.memory_system:
                return self._run_validation_checks()
            return "Click 'Refresh Validation' to run checks"
    
    def _render_graph_tab(self) -> html.Div:
        """Render semantic graph visualization tab"""
        
        graph_elements = self._build_graph_elements()
        
        return html.Div([
            html.Div([
                html.Div([
                    html.H3("Semantic Graph Memory"),
                    cyto.Cytoscape(
                        id='semantic-graph',
                        layout={'name': 'cose', 'animate': True},
                        style={'width': '100%', 'height': '500px'},
                        elements=graph_elements,
                        stylesheet=[
                            {
                                'selector': 'node',
                                'style': {
                                    'content': 'data(label)',
                                    'text-valign': 'center',
                                    'color': 'white',
                                    'text-outline-width': 2,
                                    'text-outline-color': '#888',
                                    'background-color': 'data(color)',
                                    'width': 'data(size)',
                                    'height': 'data(size)'
                                }
                            },
                            {
                                'selector': 'edge',
                                'style': {
                                    'curve-style': 'bezier',
                                    'target-arrow-shape': 'triangle',
                                    'content': 'data(label)',
                                    'line-color': '#9dbaea',
                                    'target-arrow-color': '#9dbaea',
                                    'text-rotation': 'autorotate'
                                }
                            }
                        ]
                    )
                ], style={'width': '70%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H4("Graph Statistics"),
                    html.Div(id='graph-stats', children=self._get_graph_stats()),
                    html.Hr(),
                    html.H4("Node Information"),
                    html.Div(id='graph-info', children="Select a node to view details")
                ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
            ])
        ])
    
    def _render_provenance_tab(self) -> html.Div:
        """Render provenance chain visualization tab"""
        
        provenance_fig = self._create_provenance_heatmap()
        chain_fig = self._create_provenance_chain_graph()
        
        return html.Div([
            html.H3("Provenance Explorer"),
            
            html.Div([
                html.Div([
                    html.H4("Provenance Heatmap"),
                    dcc.Graph(figure=provenance_fig)
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H4("Update Chain Graph"),
                    dcc.Graph(figure=chain_fig)
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.H4("Provenance Table"),
                dash_table.DataTable(
                    id='provenance-table',
                    data=self._get_provenance_table_data(),
                    columns=[
                        {"name": "Entry ID", "id": "entry_id"},
                        {"name": "Content", "id": "content"},
                        {"name": "Source Turn", "id": "source_turn"},
                        {"name": "Confidence", "id": "confidence"},
                        {"name": "Last Updated", "id": "last_updated"}
                    ],
                    style_cell={'textAlign': 'left'},
                    style_data={'whiteSpace': 'normal', 'height': 'auto'},
                    page_size=10
                )
            ])
        ])
    
    def _render_trace_tab(self) -> html.Div:
        """Render trace replay visualization tab"""
        
        if not self.trace_data:
            self._load_trace_data()
        
        trace_timeline = self._create_trace_timeline()
        
        return html.Div([
            html.H3("Agent Trace Replay"),
            
            html.Div([
                html.Div([
                    html.H4("Trace Timeline"),
                    dcc.Graph(figure=trace_timeline)
                ], style={'width': '100%'}),
                
                html.Div([
                    html.H4("Trace Navigation"),
                    dcc.Slider(
                        id='trace-slider',
                        min=0,
                        max=max(0, len(self.trace_data) - 1),
                        step=1,
                        value=0,
                        marks={i: f"T{i}" for i in range(0, len(self.trace_data), max(1, len(self.trace_data) // 10))}
                    )
                ], style={'margin': '20px 0'}),
                
                html.Div([
                    html.H4("Trace Details"),
                    html.Div(id='trace-details')
                ])
            ])
        ])
    
    def _render_validation_tab(self) -> html.Div:
        """Render validation and CI hooks tab"""
        
        return html.Div([
            html.H3("Validation & CI Hooks"),
            
            html.Div([
                html.Button('Refresh Validation', id='refresh-validation', n_clicks=0,
                           style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 'padding': '10px 20px'})
            ], style={'margin': '20px 0'}),
            
            html.Div(id='validation-results', children=self._run_validation_checks()),
            
            html.Div([
                html.H4("CI Hook Status"),
                html.Div([
                    html.Div([
                        html.H5("validate_graph_state()"),
                        html.P("✅ Checks for disconnected nodes and cycles"),
                        html.P("Status: Active")
                    ], className='validation-hook'),
                    
                    html.Div([
                        html.H5("check_provenance_integrity()"),
                        html.P("✅ Verifies source/update chains"),
                        html.P("Status: Active")
                    ], className='validation-hook'),
                    
                    html.Div([
                        html.H5("replay_trace(epoch)"),
                        html.P("✅ Reconstructs agent decisions"),
                        html.P("Status: Active")
                    ], className='validation-hook')
                ])
            ])
        ])
    
    def _render_rl_tab(self) -> html.Div:
        """Render RL training visualization tab"""
        
        training_fig = self._create_training_progress_chart()
        reward_fig = self._create_reward_distribution()
        
        return html.Div([
            html.H3("RL Training Dashboard"),
            
            html.Div([
                html.Div([
                    html.H4("Training Progress"),
                    dcc.Graph(figure=training_fig)
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H4("Reward Distribution"),
                    dcc.Graph(figure=reward_fig)
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.H4("Environment Statistics"),
                html.Div(id='rl-stats', children=self._get_rl_stats())
            ])
        ])
    
    def _render_status_tab(self) -> html.Div:
        """Render system status tab"""
        
        return html.Div([
            html.H3("System Status"),
            
            html.Div([
                html.Div([
                    html.H4("Memory System"),
                    html.Div(self._get_memory_system_status())
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H4("Graph Environment"),
                    html.Div(self._get_graph_env_status())
                ], style={'width': '50%', 'display': 'inline-block'})
            ])
        ])
    
    def _build_graph_elements(self) -> List[Dict[str, Any]]:
        """Build graph elements for Cytoscape visualization"""
        
        elements = []
        
        if not self.memory_system:
            # Sample graph for demo
            elements = [
                {'data': {'id': 'Paris', 'label': 'Paris', 'color': '#e74c3c', 'size': 50}},
                {'data': {'id': 'France', 'label': 'France', 'color': '#3498db', 'size': 40}},
                {'data': {'id': 'Europe', 'label': 'Europe', 'color': '#2ecc71', 'size': 60}},
                {'data': {'source': 'Paris', 'target': 'France', 'label': 'capital_of'}},
                {'data': {'source': 'France', 'target': 'Europe', 'label': 'located_in'}}
            ]
        else:
            # Build from actual memory system
            try:
                graph = self.memory_system.graph_memory.graph
                
                # Add nodes
                for node_id, node_data in graph.nodes(data=True):
                    elements.append({
                        'data': {
                            'id': node_id,
                            'label': node_id,
                            'color': '#3498db',
                            'size': 30 + node_data.get('confidence', 0.5) * 20,
                            'node_type': node_data.get('entity_type', 'entity'),
                            'confidence': node_data.get('confidence', 0.0),
                            'first_seen': node_data.get('first_seen', 'Unknown')
                        }
                    })
                
                # Add edges
                for source, target, edge_data in graph.edges(data=True):
                    elements.append({
                        'data': {
                            'source': source,
                            'target': target,
                            'label': edge_data.get('relation', 'related_to')
                        }
                    })
            
            except Exception as e:
                print(f"Error building graph elements: {e}")
        
        return elements
    
    def _get_graph_stats(self) -> html.Div:
        """Get graph statistics"""
        
        if not self.memory_system:
            return html.Div([
                html.P("Nodes: 3"),
                html.P("Edges: 2"),
                html.P("Fragments: 1"),
                html.P("Entities: 3")
            ])
        
        try:
            status = self.memory_system.get_system_status()
            graph_stats = status["module_status"]["graph_memory"]
            
            return html.Div([
                html.P(f"Nodes: {graph_stats['nodes']}"),
                html.P(f"Edges: {graph_stats['edges']}"),
                html.P(f"Fragments: {graph_stats['fragments']}"),
                html.P(f"Entities: {len(self.memory_system.graph_memory.entity_index)}")
            ])
        
        except Exception as e:
            return html.Div([html.P(f"Error loading stats: {e}")])
    
    def _create_provenance_heatmap(self) -> go.Figure:
        """Create provenance tracking heatmap"""
        
        # Sample data for demo
        data = np.random.random((10, 10))
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title="Provenance Update Frequency",
            xaxis_title="Time Steps",
            yaxis_title="Memory Entries"
        )
        
        return fig
    
    def _create_provenance_chain_graph(self) -> go.Figure:
        """Create provenance chain visualization"""
        
        # Sample network graph
        fig = go.Figure()
        
        # Add sample nodes and edges
        x_nodes = [0, 1, 2, 1]
        y_nodes = [0, 1, 0, -1]
        
        fig.add_trace(go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers+text',
            text=['Entry 1', 'Entry 2', 'Entry 3', 'Entry 4'],
            textposition="middle center",
            marker=dict(size=20, color='lightblue')
        ))
        
        # Add edges
        for i in range(len(x_nodes) - 1):
            fig.add_trace(go.Scatter(
                x=[x_nodes[i], x_nodes[i+1]], 
                y=[y_nodes[i], y_nodes[i+1]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
        
        fig.update_layout(
            title="Provenance Chain Network",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def _get_provenance_table_data(self) -> List[Dict[str, Any]]:
        """Get provenance table data"""
        
        # Sample data
        return [
            {
                "entry_id": "entry_001",
                "content": "Paris is the capital of France",
                "source_turn": 1,
                "confidence": 0.9,
                "last_updated": "2024-01-01 10:00:00"
            },
            {
                "entry_id": "entry_002", 
                "content": "France is in Europe",
                "source_turn": 2,
                "confidence": 0.8,
                "last_updated": "2024-01-01 10:01:00"
            }
        ]
    
    def _load_trace_data(self):
        """Load trace data from memory system or files"""
        
        if self.memory_system:
            try:
                # Get traces from memory system
                traces = self.memory_system.trace_buffer.get_recent_traces(50)
                self.trace_data = [asdict(trace) for trace in traces]
            except Exception as e:
                print(f"Error loading traces: {e}")
                self.trace_data = []
        else:
            # Load from file if available
            trace_files = list(Path("logs").glob("trace_*.json")) if Path("logs").exists() else []
            if trace_files:
                try:
                    with open(trace_files[0]) as f:
                        self.trace_data = json.load(f)
                except Exception as e:
                    print(f"Error loading trace file: {e}")
                    self.trace_data = []
    
    def _create_trace_timeline(self) -> go.Figure:
        """Create trace timeline visualization"""
        
        if not self.trace_data:
            return go.Figure().add_annotation(text="No trace data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        fig = go.Figure()
        
        # Extract timeline data
        turns = [i for i in range(len(self.trace_data))]
        rewards = [trace.get('reward_signal', 0) for trace in self.trace_data]
        
        fig.add_trace(go.Scatter(
            x=turns,
            y=rewards,
            mode='lines+markers',
            name='Reward Signal',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Agent Trace Timeline",
            xaxis_title="Turn",
            yaxis_title="Reward Signal"
        )
        
        return fig
    
    def _run_validation_checks(self) -> html.Div:
        """Run validation checks and return results"""
        
        if not self.memory_system:
            return html.Div([
                html.H4("Validation Results"),
                html.P("⚠️ Memory system not available for validation")
            ])
        
        try:
            # Run validation hooks
            graph_validation = self.memory_system.validate_graph_state()
            provenance_validation = self.memory_system.check_provenance_integrity()
            
            # Try trace replay if possible
            replay_result = None
            if self.memory_system.current_turn >= 2:
                replay_result = self.memory_system.replay_trace(0, min(2, self.memory_system.current_turn))
            
            return html.Div([
                html.H4("Validation Results"),
                
                html.Div([
                    html.H5("Graph State Validation"),
                    html.P(f"Status: {graph_validation.get('overall_status', 'Unknown')}"),
                    html.P(f"Disconnected nodes: {graph_validation.get('disconnected_nodes', 0)}"),
                    html.P(f"Cycles detected: {graph_validation.get('cycles_detected', 0)}")
                ]),
                
                html.Div([
                    html.H5("Provenance Integrity"),
                    html.P(f"Status: {provenance_validation.get('overall_status', 'Unknown')}"),
                    html.P(f"Broken chains: {provenance_validation.get('broken_chains', 0)}"),
                    html.P(f"Orphaned entries: {provenance_validation.get('orphaned_entries', 0)}")
                ]),
                
                html.Div([
                    html.H5("Trace Replay"),
                    html.P(f"Status: {'✅ Success' if replay_result and 'error' not in replay_result else '⚠️ Not available'}"),
                    html.P(f"Traces replayed: {replay_result.get('base_replay', {}).get('traces_replayed', 0) if replay_result else 0}")
                ])
            ])
        
        except Exception as e:
            return html.Div([
                html.H4("Validation Results"),
                html.P(f"❌ Error running validation: {e}")
            ])
    
    def _create_training_progress_chart(self) -> go.Figure:
        """Create RL training progress chart"""
        
        if not self.graph_env:
            return go.Figure().add_annotation(text="Graph environment not available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        try:
            env_stats = self.graph_env.get_environment_stats()
            trace_buffer = self.graph_env.get_trace_buffer()
            
            if not trace_buffer:
                return go.Figure().add_annotation(text="No training data available", 
                                                xref="paper", yref="paper", x=0.5, y=0.5)
            
            episodes = [ep["episode"] for ep in trace_buffer]
            rewards = [ep["episode_reward"] for ep in trace_buffer]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=episodes,
                y=rewards,
                mode='lines+markers',
                name='Episode Reward'
            ))
            
            fig.update_layout(
                title="RL Training Progress",
                xaxis_title="Episode",
                yaxis_title="Episode Reward"
            )
            
            return fig
        
        except Exception as e:
            return go.Figure().add_annotation(text=f"Error: {e}", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
    
    def _create_reward_distribution(self) -> go.Figure:
        """Create reward distribution chart"""
        
        # Sample reward distribution
        rewards = np.random.normal(0.5, 0.2, 100)
        
        fig = go.Figure(data=[go.Histogram(x=rewards, nbinsx=20)])
        fig.update_layout(
            title="Reward Distribution",
            xaxis_title="Reward Value",
            yaxis_title="Frequency"
        )
        
        return fig
    
    def _get_rl_stats(self) -> html.Div:
        """Get RL training statistics"""
        
        if not self.graph_env:
            return html.Div([html.P("Graph environment not available")])
        
        try:
            stats = self.graph_env.get_environment_stats()
            
            return html.Div([
                html.P(f"Current Episode: {stats['environment_info']['current_episode']}"),
                html.P(f"Current Turn: {stats['environment_info']['current_turn']}"),
                html.P(f"Total Episodes: {stats['trace_statistics']['total_episodes']}"),
                html.P(f"Average Episode Reward: {stats['reward_statistics']['avg_episode_reward']:.3f}")
            ])
        
        except Exception as e:
            return html.Div([html.P(f"Error loading RL stats: {e}")])
    
    def _get_memory_system_status(self) -> html.Div:
        """Get memory system status"""
        
        if not self.memory_system:
            return html.Div([html.P("❌ Memory system not connected")])
        
        try:
            status = self.memory_system.get_system_status()
            
            return html.Div([
                html.P(f"✅ Memory system active"),
                html.P(f"Current turn: {status['system_stats']['current_turn']}"),
                html.P(f"Total extractions: {status['system_stats']['total_extractions']}"),
                html.P(f"Total operations: {status['system_stats']['total_operations']}")
            ])
        
        except Exception as e:
            return html.Div([html.P(f"❌ Error: {e}")])
    
    def _get_graph_env_status(self) -> html.Div:
        """Get graph environment status"""
        
        if not self.graph_env:
            return html.Div([html.P("❌ Graph environment not connected")])
        
        try:
            stats = self.graph_env.get_environment_stats()
            
            return html.Div([
                html.P(f"✅ Graph environment active"),
                html.P(f"Dataset size: {stats['environment_info']['dataset_size']}"),
                html.P(f"Max turns: {stats['environment_info']['max_turns']}"),
                html.P(f"Total traces: {stats['trace_statistics']['total_traces']}")
            ])
        
        except Exception as e:
            return html.Div([html.P(f"❌ Error: {e}")])
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """Run the dashboard server"""
        print(f"🚀 Starting Memory-R1 Dashboard on http://localhost:{port}")
        self.app.run_server(debug=debug, port=port)

# Utility functions
def create_dashboard(memory_system=None, graph_env=None) -> MemoryR1Dashboard:
    """Create Memory-R1 dashboard instance"""
    return MemoryR1Dashboard(memory_system, graph_env)

def run_dashboard_demo():
    """Run dashboard demo with sample data"""
    
    print("📊 Memory-R1 Dashboard Demo")
    
    # Create dashboard without systems for demo
    dashboard = create_dashboard()
    
    # Run server
    dashboard.run_server(debug=True, port=8050)

if __name__ == '__main__':
    run_dashboard_demo()
