import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from castle.algorithms import PC, Notears
import matplotlib.pyplot as plt

st.title('Simple Causal Discovery')
st.caption('Assuming linear, independently and individually distributed data')

upload = st.file_uploader("Upload a CSV file", type='csv')
if upload is not None:
    data = pd.read_csv(upload)

textinput = st.text_input('put in the variable names that you would like to test')
st.caption('please use comma to separate the names, with no spaces before and after the comma such as "x,y" or "a,b"')

rad = st.radio("select causal discovery algorithm", ('PC', 'NOTEARS'))
st.caption("please select the NOTEARS option for the continuous optimization method")

process = st.button('run causal inference')

if process:
    textinput = textinput.split(',')

    count = len(textinput)

    st.dataframe(data)

    data = data[data.columns.intersection(textinput)]

    matrix = data.to_numpy()

    pc = PC()

    notears = Notears()



    if rad == 'PC':
        pc.learn(matrix)
        graph = nx.DiGraph(pc.causal_matrix)
    else:
        notears.learn(matrix)
        graph = nx.DiGraph(notears.causal_matrix)

    cols = data.columns.values.tolist()

    MAPPING = {k: v for k, v in zip(range(count), cols)}
    graph = nx.relabel_nodes(graph, MAPPING, copy=True)

    fig, ax = plt.subplots()
    nx.draw(graph, font_size = 5, node_size = 1800, font_color = 'white', with_labels=True)
    st.pyplot(fig)

st.text(
    "results may not reflect true causal relationship. ")
st.text(
    "It is recommended to check results with domain knowledge and additional tests")
