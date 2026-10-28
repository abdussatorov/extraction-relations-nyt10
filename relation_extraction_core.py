import networkx as nx
import pandas as pd

ENT_TYPES = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC"}

def build_graph(doc):
    g = nx.Graph()
    for tok in doc:
        g.add_node(tok.i)
        if tok.head.i != tok.i:
            g.add_edge(tok.i, tok.head.i)
    return g

def get_type(doc, span):
    for ent in doc.ents:
        if ent.start < span.end and span.start < ent.end:
            return ent.label_
    return "NONE"

def dep_path(doc, g, h, t):
    a, b = h.root.i, t.root.i
    if a == b:
        return None
    try:
        nodes = nx.shortest_path(g, a, b)
    except nx.NetworkXNoPath:
        return None
    steps = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        if doc[u].head.i == v:
            steps.append(f"UP_{doc[u].dep_}")
        else:
            steps.append(f"DOWN_{doc[v].dep_}")
    inner = [doc[i].lemma_.lower() for i in nodes[1:-1]]
    return steps, inner

def pair_features(doc, g, h, t):
    p = dep_path(doc, g, h, t)
    if p is None:
        steps, inner = ["NOPATH"], []
    else:
        steps, inner = p
    h_first = h.start < t.start
    first, second = (h, t) if h_first else (t, h)
    between = doc[first.end:second.start]
    between_txt = " ".join(tok.lemma_.lower() for tok in between if not tok.is_space and not tok.is_punct)
    h_type, t_type = get_type(doc, h), get_type(doc, t)
    path_tokens = steps + ["W_" + w for w in inner] + ["PATH_" + "_".join(steps)]
    return dict(
        between=between_txt, path=" ".join(path_tokens),
        h_type=h_type, t_type=t_type, type_pair=f"{h_type}>{t_type}",
        order="HT" if h_first else "TH", dist=len(between), path_len=len(steps)
    )

def extract_triples(text, nlp, model, threshold=0.5):
    doc = nlp(text)
    graph = build_graph(doc)
    candidates, rows = [], []
    for sent in doc.sents:
        ents = [e for e in sent.ents if e.label_ in ENT_TYPES]
        for h in ents:
            for t in ents:
                if h.start == t.start:
                    continue
                candidates.append((h, t, sent.text.strip()))
                rows.append(pair_features(doc, graph, h, t))
    if not rows:
        return [], doc
    probs = model.predict_proba(pd.DataFrame(rows))
    classes = model.classes_
    triples = []
    for (h, t, sent), prob in zip(candidates, probs):
        idx = prob.argmax()
        relation = classes[idx]
        confidence = float(prob[idx])
        if relation != "NA" and confidence >= threshold:
            triples.append({
                "h": h.text, "relation": relation, "t": t.text,
                "proba": round(confidence, 3), "phrase": sent
            })
    # Remove duplicate triplets, keeping the highest confidence.
    unique = {}
    for item in triples:
        key = (item["h"], item["relation"], item["t"])
        if key not in unique or item["proba"] > unique[key]["proba"]:
            unique[key] = item
    return sorted(unique.values(), key=lambda x: x["proba"], reverse=True), doc
