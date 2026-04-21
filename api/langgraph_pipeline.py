from typing import TypedDict, List, Dict
import requests
import logging
import os
import yaml
from langgraph.graph import StateGraph, END

# Load config
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

# Logging
log_file = config['logging']['file']
if not os.path.isabs(log_file):
    log_file = os.path.join(os.path.dirname(__file__), '..', log_file)
log_dir = os.path.dirname(log_file)
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=config['logging']['level'], filename=log_file, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('langgraph_pipeline')


class AgentState(TypedDict):
    user_input: str
    planner_decision: str
    risk_categories: List[str]
    rewritten_prompt: str
    generated_response: str
    verification: Dict
    retry_count: int


def call_agent(url, payload, timeout=600):
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Agent call failed {url} payload={payload} error={e}")
        return None


def call_planner(user_input: str):
    planner_url = f"http://localhost:{config['ports']['planner']}/planner/analyze"
    return call_agent(planner_url, {"prompt": user_input}) or {
        'decision': 'BLOCK',
        'categories': [],
        'risk_score': 1.0,
        'reason': 'planner failure'
    }


def call_researcher(user_input: str, planner_decision: str = "", categories: List[str] | None = None):
    researcher_url = f"http://localhost:{config['ports']['researcher']}/researcher/rewrite"
    return call_agent(
        researcher_url,
        {
            "prompt": user_input,
            "planner_decision": planner_decision,
            "categories": categories or [],
        },
    ) or {
        'rewritten_prompt': ''
    }


def call_generator(prompt: str):
    generator_url = f"http://localhost:{config['ports']['generator']}/generator/respond"
    return call_agent(generator_url, {"prompt": prompt}) or {
        'response': "I'm sorry, an error occurred while generating the response."
    }


def call_verifier(user_input: str, rewritten_prompt: str, generated_response: str):
    verifier_url = f"http://localhost:{config['ports']['verifier']}/verifier/check"
    payload = {
        'prompt': rewritten_prompt or user_input,
        'response': generated_response
    }
    return call_agent(verifier_url, payload) or {
        'verdict': 'REJECT',
        'faithfulness_score': 0.0,
        'relevance_score': 0.0,
        'hallucination_flags': ['verifier unavailable'],
        'verified_response': ''
    }


def planner_node(state: AgentState):
    result = call_planner(state['user_input'])
    state['planner_decision'] = result.get('decision', 'BLOCK')
    state['risk_categories'] = result.get('categories', [])
    logger.info(f"planner_node: decision={state['planner_decision']} categories={state['risk_categories']}")
    return {
        'planner_decision': state['planner_decision'],
        'risk_categories': state['risk_categories']
    }


def researcher_node(state: AgentState):
    result = call_researcher(
        state['user_input'],
        planner_decision=state.get('planner_decision', ''),
        categories=state.get('risk_categories', []),
    )
    state['rewritten_prompt'] = result.get('rewritten_prompt', '')
    logger.info(f"researcher_node: rewritten_prompt={state['rewritten_prompt']}")
    return {'rewritten_prompt': state['rewritten_prompt']}


def generator_node(state: AgentState):
    prompt = state.get('rewritten_prompt') or state['user_input']
    result = call_generator(prompt)
    state['generated_response'] = result.get('response', '')
    logger.info(f"generator_node: generated_response={state['generated_response']}")
    return {'generated_response': state['generated_response']}


def verifier_node(state: AgentState):
    result = call_verifier(state['user_input'], state.get('rewritten_prompt', ''), state['generated_response'])
    state['verification'] = result
    logger.info(f"verifier_node: verification={result}")
    return {'verification': state['verification']}


def retry_node(state: AgentState):
    state['retry_count'] = state.get('retry_count', 0) + 1
    logger.info(f"retry_node: retry_count={state['retry_count']}")
    return {'retry_count': state['retry_count']}


def planner_router(state: AgentState):
    decision = state['planner_decision']
    if decision == 'ALLOW':
        return 'generator'
    elif decision == 'REWRITE':
        return 'researcher'
    else:
        return END


def verifier_router(state: AgentState):
    verdict = state['verification'].get('verdict', 'REJECT')
    if verdict == 'PASS':
        return END
    elif verdict == 'RETRY' and state.get('retry_count', 0) < 2:
        return 'retry'
    else:
        return END


# Build graph
builder = StateGraph(AgentState)
builder.add_node('planner', planner_node)
builder.add_node('researcher', researcher_node)
builder.add_node('generator', generator_node)
builder.add_node('verifier', verifier_node)
builder.add_node('retry', retry_node)

builder.set_entry_point('planner')

builder.add_conditional_edges('planner', planner_router, {
    'generator': 'generator',
    'researcher': 'researcher',
    END: END
})

builder.add_edge('researcher', 'generator')
builder.add_edge('generator', 'verifier')

builder.add_conditional_edges('verifier', verifier_router, {
    END: END,
    'retry': 'retry'
})

builder.add_edge('retry', 'generator')

graph = builder.compile()

# Graph visualization
# Ensure Graphviz executables are on PATH for this process (Windows path default location)
graphviz_bin = r"C:\Program Files\Graphviz\bin"
if graphviz_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = os.environ.get('PATH', '') + os.pathsep + graphviz_bin

try:
    from graphviz import Digraph

    dot = Digraph(name='LLMINTD_LangGraph', format='png')
    dot.attr(rankdir='LR')
    dot.attr('node', shape='box')

    nodes = ['User Input', 'Planner', 'Researcher', 'Generator', 'Verifier', 'Retry', 'End_Block', 'End_Pass', 'End_Fail']
    for n in nodes:
        dot.node(n)

    edges = [
        ('User Input', 'Planner'),
        ('Planner', 'Generator', 'ALLOW'),
        ('Planner', 'Researcher', 'REWRITE'),
        ('Planner', 'End_Block', 'BLOCK'),
        ('Researcher', 'Generator', ''),
        ('Generator', 'Verifier', ''),
        ('Verifier', 'End_Pass', 'PASS'),
        ('Verifier', 'Retry', 'RETRY'),
        ('Verifier', 'End_Fail', 'REJECT'),
        ('Retry', 'Generator', '')
    ]

    for edge in edges:
        if len(edge) == 3:
            dot.edge(edge[0], edge[1], label=edge[2])
        else:
            dot.edge(edge[0], edge[1])

    dot.render(filename='langgraph_pipeline', cleanup=True)
    print('Graph visualization saved as langgraph_pipeline.png (and optional PDF/SVG)')
except Exception as e:
    print(f'Graphviz Python render failed: {e}')


def run_pipeline(user_input: str) -> Dict:
    initial_state: AgentState = {
        'user_input': user_input,
        'planner_decision': '',
        'risk_categories': [],
        'rewritten_prompt': '',
        'generated_response': '',
        'verification': {},
        'retry_count': 0
    }
    logger.info(f"run_pipeline: input={user_input}")

    result_state = graph.invoke(initial_state)
    logger.info(f"run_pipeline: final_state={result_state}")

    return {
        'final_response': result_state.get('generated_response', ''),
        'verification': result_state.get('verification', {}),
        'retry_count': result_state.get('retry_count', 0)
    }
