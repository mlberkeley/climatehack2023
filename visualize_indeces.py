import json

import matplotlib.pyplot as plt
with open("indices.json") as f:
	site_locations = {
		data_source: {
			int(site): (int(location[0]), int(location[1]))
			for site, location in locations.items()
                        }
		for data_source, locations in json.load(f).items()
                 }

print([key for key in site_locations])
color_dict = {'hrv': 'blue', 'nonhrv': 'red', 'weather': 'orange', 'aerosols': 'grey'}

for k in site_locations:
	site = site_locations[k]
	x_values, y_values = [site[key][0] for key in site], [site[key][1] for key in site]

	# Plot the points
	plt.scatter(x_values, y_values, color=color_dict[k], marker='o', label=k)

# Add labels and a title
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Scatter Plot of Points')

# Show the legend
plt.legend()

# Show the plot
plt.show()
