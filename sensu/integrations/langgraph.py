"""
LangGraph callback handler for Sensu telemetry.

LangGraph builds on LangChain Core's callback system, so the existing
``SensuCallbackHandler`` already captures LLM calls, tool calls, and step
boundaries inside a graph. This handler is a thin subclass that surfaces a
discoverable import path for LangGraph users and identifies itself as such
in LangChain's debug output. The actual ``langgraph_node`` detection (which
emits ``step_type='langgraph_node'`` with the node name) lives in the shared
LangChain handler so it works regardless of which class the customer
instantiates — including for mixed LangChain + LangGraph projects.

Usage::

    import sensu
    from sensu.integrations.langgraph import SensuLangGraphHandler
    from langgraph.graph import StateGraph

    client = sensu.SensuClient({"from_env": True})
    handler = SensuLangGraphHandler(client=client)

    graph = StateGraph(MyState).add_node(...).compile()
    result = await graph.ainvoke(inputs, config={"callbacks": [handler]})

Requires the ``langgraph`` extra::

    pip install 'sensu-sdk[langgraph]'

For mixed LangChain + LangGraph projects, a single ``SensuLangGraphHandler``
captures both: non-graph chains emit ``step_type='chain'``, graph nodes
emit ``step_type='langgraph_node'`` with the node name.
"""
from __future__ import annotations

from sensu.integrations.langchain import SensuCallbackHandler


class SensuLangGraphHandler(SensuCallbackHandler):
    """Thin subclass of SensuCallbackHandler with a LangGraph-specific name."""

    # Surfaced in LangChain's debug output and trace dumps.
    name = "sensu_langgraph_handler"


__all__ = ["SensuLangGraphHandler"]
