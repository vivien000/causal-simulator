import numpy as np
import pandas as pd
from sklearn import linear_model
import streamlit as st
from st_cytoscape import cytoscape

num_observations = 10_000
generating_model = """
Z <-- N
C1 <-- N
C21 <-- N
C31 <-- N
C33 <-- N
C22 <-- C21 + N
C32 <-- C31 + C33 + N
X <-- Z + C1 + C21 + C31 + N
M <-- X + N
Y <-- M + C1 + C22 + C33 + N
C4 <-- X + Y + N
"""


def rewrite(x):
    if x == "N":
        return "np.random.randn(num_observations)"
    else:
        return f'd["{x}"]'


@st.cache()
def generate_data():
    np.random.seed(seed=0)
    d = {}
    nodes = set()
    edges = set()
    for line in generating_model.split("\n"):
        if " <-- " in line:
            left, right = line.split(" <-- ")
            right_terms = right.split(" + ")
            nodes.add(left)
            for node in right_terms:
                if node != "N":
                    nodes.add(node)
                    edges.add((node, left))
            formula = f"{rewrite(left)} = {' + '.join(list(map(rewrite, right_terms)))}"
            exec(formula)
    return pd.DataFrame.from_dict(d), nodes, edges


df, nodes, edges = generate_data()
elements = []
for node in nodes:
    elements.append(
        {
            "data": {"id": node},
            "selected": node == "X",
            "selectable": node not in ["X", "Y"],
        }
    )
for edge in edges:
    elements.append(
        {
            "data": {
                "source": edge[0],
                "target": edge[1],
                "id": f"{edge[0]}-{edge[1]}",
            },
            "selectable": False,
        }
    )
stylesheet = [
    {"selector": "node", "style": {"label": "data(id)", "width": 20, "height": 20}},
    {
        "selector": "edge",
        "style": {
            "width": 2,
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
        },
    },
]

layout = {"name": "fcose", "animationDuration": 0}
layout["alignmentConstraint"] = {"horizontal": [["Z", "X", "M", "Y"]]}
layout["relativePlacementConstraint"] = [{"left": "X", "right": "Y"}]
layout["relativePlacementConstraint"].append({"top": "C1", "bottom": "X"})
layout["relativePlacementConstraint"].append({"top": "C21", "bottom": "X"})
layout["relativePlacementConstraint"].append({"top": "X", "bottom": "C4"})
layout["relativePlacementConstraint"].append({"top": "X", "bottom": "C31"})
layout["nodeRepulsion"] = 50000

st.sidebar.title("Causal simulator")

st.sidebar.markdown(
    """
**Estimating the effect of a variable X** (e.g. vaccination status) **on another variable Y** (e.g. symptoms) **may require controlling for other variables** (e.g. age if age increases both access to the vaccine and the risks of getting sick).

This demo illustrates that **the choice of the variables to control for critically depends on the causal relationships between the variables**.

*Inspired by [The Book of Why](http://bayes.cs.ucla.edu/WHY/) by Judea Pearl and Dana Mackenzie and built by [Vivien](https://twitter.com/vivien000000) with [Streamlit](https://streamlit.io/), [Cytoscape.js](https://js.cytoscape.org/) and [scikit-learn](https://scikit-learn.org/stable/)*
"""
)

st.subheader("Data generating process")

st.markdown(
    """
10,000 observations have been generated for the variables mentioned in the causal graph below. The values for each variable were derived from the values of its parents in the causal graph as follows:
"""
)
st.latex(
    "U = \sum_{V \in \mathrm{\ Parents}(U)} V + \epsilon_U  \quad \mathrm{where} \quad \epsilon_U \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, 1)"
)
st.subheader("Results of controlling for certain variables")

st.markdown(
    "We are using a linear regression to estimate the effect on Y of increasing X by one unit. In the causal graph below, **select the variables to include in the regression and see whether the regression coefficient for X matches the expected value (1, given the data generating process)**."
)


def add_smiley(x):
    return x + (" 😀" if np.abs(float(x) - 1) < 0.1 else " 😨")


col1, col2 = st.columns(2)
order = ["X"] + sorted([n for n in nodes if n not in ["X", "Y"]])
results = {v: "" for v in order}
with col1:
    selected = cytoscape(
        elements,
        stylesheet,
        height="450px",
        layout=layout,
        selection_type="additive",
        user_panning_enabled=False,
        user_zooming_enabled=False,
        key="graph",
    )
try:
    selected_nodes = [n for n in order if n in selected["nodes"]]
    regr = linear_model.LinearRegression()
    regr.fit(df[selected_nodes], df[["Y"]])
    for i in range(len(regr.feature_names_in_)):
        results[regr.feature_names_in_[i]] = "%.3f" % regr.coef_[0, i]
    results["X"] = add_smiley(results["X"])
    with col2:
        table = "<table style='margin: auto;'><tbody>"
        table += "<tr><td colspan=2><b>Regression coefficients</b></td></tr>"
        for k in order:
            table += f"<tr><td>{k}</td><td>{results[k]}</td></tr>"
        table += "</table></tbody>"
        st.markdown(table, unsafe_allow_html=True)
except TypeError:
    pass


@st.cache()
def compute_instrumental_variable():
    regr = linear_model.LinearRegression()
    regr2 = linear_model.LinearRegression()
    regr.fit(df[["Z"]], df[["Y"]])
    regr2.fit(df[["Z"]], df[["X"]])
    return "%.3f" % (regr.coef_[0, 0] / regr2.coef_[0, 0])


@st.cache()
def compute_front_door():
    regr = linear_model.LinearRegression()
    regr2 = linear_model.LinearRegression()
    regr.fit(df[["M", "X"]], df[["Y"]])
    regr2.fit(df[["X"]], df[["M"]])
    index_m = [i for i in range(2) if regr.feature_names_in_[i] == "M"][0]
    return "%.3f" % (regr.coef_[0, index_m] * regr2.coef_[0, 0])


result_instrumental_variable = add_smiley(compute_instrumental_variable())
result_front_door = add_smiley(compute_front_door())

with st.expander("What variables should be controlled for?"):
    st.markdown(
        """The ***back-door criterion*** (*Book of Why*, chapter 4) provides sufficient conditions for variables to be adequate controls:
- The non-causal paths between X and Y going through C1, C21 and C22 need to be blocked by controlling for **C1**, as well as for **C21 or C22** or both;
- C4 and C32 are *colliders* (common consequences of their two respective neighbors). The 2 non-causal paths going through them are then blocked as long as C4 and C32 are not controlled for. Therefore, C4, C31, C32 and C33 should not be controlled for. However, if C32 is controlled for, C31 or C33 or both should be controlled for to block the non-causal path now open;
- M is on a causal path from X to Y and should not be controlled for (so that the causal effect of X on Y is not masked);
- Z is only a cause of X. Controlling for it is useless.
"""
    )
with st.expander("Bonus: front-door criterion and instrumental variable"):
    st.markdown(
        f"""
If using the back-door criterion as described above is impossible, we can deduce from the causal diagram that two alternative methods are available:
- the use of the ***instrumental variable*** Z (result: {result_instrumental_variable})
- the ***front-door criterion*** (*Book of Why*, chapter 5) with M as a mediator (result: {result_front_door})
"""
    )
