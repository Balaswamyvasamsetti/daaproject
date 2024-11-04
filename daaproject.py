import customtkinter as ctk
import heapq
import matplotlib.pyplot as plt
import networkx as nx
from tkinter import messagebox
from matplotlib.animation import FuncAnimation

# Initialize CustomTkinter theme and appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class Graph:
    def __init__(self):
        self.graph = {}
    def add_edge(self, from_node, to_node, weight):
        if from_node not in self.graph:
            self.graph[from_node] = []
        self.graph[from_node].append((to_node, weight))
    def remove_edge(self, from_node, to_node):
        if from_node in self.graph:
            self.graph[from_node] = [edge for edge in self.graph[from_node] if edge[0] != to_node]
        if to_node in self.graph:
            self.graph[to_node] = [edge for edge in self.graph[to_node] if edge[0] != from_node]

    def dijkstra(self, start, end):
        queue = [(0, start)]
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0
        parent = {start: None}
        while queue:
            current_distance, current_node = heapq.heappop(queue)
            if current_node == end:
                break
            if current_distance > distances[current_node]:
                continue
            for neighbor, weight in self.graph[current_node]:
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    parent[neighbor] = current_node
                    heapq.heappush(queue, (distance, neighbor))
        path = []
        while end is not None:
            path.append(end)
            end = parent[end]
        path.reverse()
        return path, distances[path[-1]] if distances[path[-1]] != float('inf') else None

class EmergencyRoutingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Emergency Vehicle Routing System")
        self.geometry("800x600")  # Adjusted width for better layout
        # Initialize graph for storing nodes and edges
        self.graph = Graph()
        # Title label
        self.title_label = ctk.CTkLabel(self, text="Emergency Vehicle Routing System", font=("Arial", 24, "bold"))
        self.title_label.pack(pady=10)
        # Create a frame for the input section
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(side="left", fill="y", padx=10, pady=10)
        # Create a frame for the output section
        self.output_frame = ctk.CTkFrame(self)
        self.output_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        # Node Input Section
        self.node_frame = ctk.CTkFrame(self.input_frame)
        self.node_frame.pack(pady=10)
        # From and To nodes
        self.start_node_label = ctk.CTkLabel(self.node_frame, text="From:", font=("Arial", 14))
        self.start_node_label.grid(row=0, column=0, padx=5, pady=5)
        self.start_node_entry = ctk.CTkEntry(self.node_frame, width=120)
        self.start_node_entry.grid(row=0, column=1, padx=5, pady=5)

        self.end_node_label = ctk.CTkLabel(self.node_frame, text="To:", font=("Arial", 14))
        self.end_node_label.grid(row=0, column=2, padx=5, pady=5)
        self.end_node_entry = ctk.CTkEntry(self.node_frame, width=120)
        self.end_node_entry.grid(row=0, column=3, padx=5, pady=5)
        # Weight (Distance) input
        self.weight_label = ctk.CTkLabel(self.input_frame, text="Distance:", font=("Arial", 14))
        self.weight_label.pack(pady=5)
        self.weight_entry = ctk.CTkEntry(self.input_frame, width=100)
        self.weight_entry.pack(pady=5)
        # Add Edge and Clear Graph buttons
        self.add_edge_button = ctk.CTkButton(self.input_frame, text="Add Route", 
                                             command=self.add_edge, fg_color="#4CAF50")
        self.add_edge_button.pack(pady=10)
        self.remove_edge_button = ctk.CTkButton(self.input_frame, text="Remove Route", 
                                                command=self.remove_edge, fg_color="#0000FF")
        self.remove_edge_button.pack(pady=10)
        self.clear_graph_button = ctk.CTkButton(self.input_frame, text="Clear Graph", 
                                                command=self.clear_graph, fg_color="#FF5252")
        self.clear_graph_button.pack(pady=10)
        # Calculate Path Section
        self.calc_frame = ctk.CTkFrame(self.input_frame)
        self.calc_frame.pack(pady=20)
        self.calc_start_label = ctk.CTkLabel(self.calc_frame, text="Start Location:", font=("Arial", 14))
        self.calc_start_label.grid(row=0, column=0, padx=5, pady=5)
        self.calc_start_entry = ctk.CTkEntry(self.calc_frame, width=120)
        self.calc_start_entry.grid(row=0, column=1, padx=5, pady=5)

        self.calc_end_label = ctk.CTkLabel(self.calc_frame, text="Destination:", font=("Arial", 14))
        self.calc_end_label.grid(row=1, column=0, padx=5, pady=5)
        self.calc_end_entry = ctk.CTkEntry(self.calc_frame, width=120)
        self.calc_end_entry.grid(row=1, column=1, padx=5, pady=5)

        self.calculate_button = ctk.CTkButton(self.input_frame, text="Calculate Shortest Route", 
                                              command=self.calculate_path)
        self.calculate_button.pack(pady=10)
        # Results Section
        self.result_label = ctk.CTkLabel(self.output_frame, text="Route Summary", font=("Arial", 16, "bold"))
        self.result_label.pack(pady=5)
        # Create separate text boxes for added distances and shortest path results
        self.added_routes_label = ctk.CTkLabel(self.output_frame, text="Added Routes", font=("Arial", 14))
        self.added_routes_label.pack(pady=5)
        self.added_routes_text = ctk.CTkTextbox(self.output_frame, width=500, height=400)
        self.added_routes_text.pack(pady=10, fill="both")

        self.shortest_path_label = ctk.CTkLabel(self.output_frame, text="Shortest Path Result", font=("Arial", 14))
        self.shortest_path_label.pack(pady=5)
        self.shortest_path_text = ctk.CTkTextbox(self.output_frame, width=500, height=100)
        self.shortest_path_text.pack(pady=10, fill="both", expand=True)

    def add_edge(self):
        from_node = self.start_node_entry.get().strip()
        to_node = self.end_node_entry.get().strip()
        weight = self.weight_entry.get().strip()
        
        try:
            weight = float(weight)
            if from_node and to_node:
                self.graph.add_edge(from_node, to_node, weight)
                self.graph.add_edge(to_node, from_node, weight)  # Assuming bidirectional
                self.added_routes_text.insert("end",f"Added route: {from_node} -> {to_node} with distance{weight} kms\n")
                self.clear_entry_fields()
                messagebox.showinfo("Success", f"Route from {from_node} to {to_node} added successfully!")
            else:
                messagebox.showwarning("Input Error", "Please fill both 'From' and 'To' fields.")
        except ValueError:
            messagebox.showwarning("Input Error", "Distance should be a valid number.")

    def remove_edge(self):
        from_node = self.start_node_entry.get().strip()
        to_node = self.end_node_entry.get().strip()

        if from_node and to_node:
            self.graph.remove_edge(from_node, to_node)
            self.added_routes_text.insert("end", f"Removed route: {from_node} <-> {to_node}\n")
            self.clear_entry_fields()
            messagebox.showinfo("Success", f"Route from {from_node} to {to_node} removed successfully!")
        else:
            messagebox.showwarning("Input Error", "Please fill both 'From' and 'To' fields.")

    def clear_graph(self):
        self.graph = Graph()
        self.added_routes_text.delete("1.0", "end")
        self.shortest_path_text.delete("1.0", "end")
        messagebox.showinfo("Success", "Graph cleared successfully!")

    def calculate_path(self):
        start = self.calc_start_entry.get().strip()
        end = self.calc_end_entry.get().strip()

        if start and end:
            path, distance = self.graph.dijkstra(start, end)
            if path:
                self.shortest_path_text.delete("1.0", "end")
                self.shortest_path_text.insert("end", f"Shortest path: {' -> '.join(path)}\nTotal distance:{distance}Kms \n")
                self.animate_path(path)
            else:
                self.shortest_path_text.delete("1.0", "end")
                messagebox.showwarning("No Path", f"No path found from {start} to {end}.")
        else:
            messagebox.showwarning("Input Error", "Please fill both 'Start' and 'Destination' fields.")

    def clear_entry_fields(self):
        self.start_node_entry.delete(0, 'end')
        self.end_node_entry.delete(0, 'end')
        self.weight_entry.delete(0, 'end')
        self.calc_start_entry.delete(0, 'end')
        self.calc_end_entry.delete(0, 'end')

    def animate_path(self, path):
        G = nx.Graph()
        for node in self.graph.graph:
            G.add_node(node)
        for node in self.graph.graph:
            for neighbor, weight in self.graph.graph[node]:
                G.add_edge(node, neighbor, weight=weight)

        pos = nx.spring_layout(G)

        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue', node_size=500, font_size=12, 
                font_weight='bold', edge_color='gray')
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        path_edges_labels = {edge: G[edge[0]][edge[1]]['weight'] for edge in path_edges}

        def update(frame):
            ax.clear()
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue', node_size=500, font_size=12, 
                    font_weight='bold', edge_color='gray')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
            if frame < len(path_edges):
                nx.draw_networkx_edges(G, pos, edgelist=path_edges[:frame + 1], edge_color='red', width=2)
                nx.draw_networkx_edge_labels(G, pos, edge_labels=path_edges_labels, font_color='red')

        ani = FuncAnimation(fig, update, frames=len(path_edges) + 1, interval=1000)
        plt.title("Animating Shortest Path")
        plt.show()

if __name__ == "__main__":
    app = EmergencyRoutingApp()
    app.mainloop()
