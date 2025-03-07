import graphviz

# Kreiraj jednostavan graf
dot = graphviz.Digraph()
dot.node('A', 'Python')
dot.node('B', 'Graphviz')
dot.edge('A', 'B')

# Sačuvaj kao PDF
dot.render('test_graph', format='pdf', cleanup=True)