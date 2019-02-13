import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Panel, CategoricalColorMapper
from bokeh.models.widgets import CheckboxGroup, Tabs, Panel
from bokeh.layouts import column, row, WidgetBox
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.layouts import row, column, gridplot

class BokehHistogram():
    '''
    A class to simplify the making of interactive histograms with the Bokeh library.
    Requires: Bokeh, Pandas, and Numpy.
    '''

    def __init__(self, colors=["SteelBlue", "Tan"], height=600, width=600):
        self.colors = colors
        self.height = height
        self.width = width

    def hist_hover(self, dataframe, column, bins=30, log_scale=False, show_plot=True):
        """
        A method for creating a sinlge Bokeh histogram with hovertool interactivity.

        Parameters:
        ----------
        Input:
        dataframe {df}: Pandas dataframe
        column {string}: column of dataframe to plot in histogram
        bins {int}: number of bins in histogram
        log_scale {bool}: True to plot on a log scale
        colors {list -> string}: list of colors for histogram; first color default color, second color is hover color
        show_plot {bool}: True to display the plot, False to store the plot in a variable (for use in later methods)

        Output:
        plot: bokeh historgram with interactive hover tool

        """
        # build histogram data with Numpy
        hist, edges = np.histogram(dataframe[column], bins = bins)
        hist_df = pd.DataFrame({column: hist,
                                 "left": edges[:-1],
                                 "right": edges[1:]})
        hist_df["interval"] = ["%d to %d" % (left, right) for left, 
                               right in zip(hist_df["left"], hist_df["right"])]
        # bokeh histogram with hover tool
        if log_scale == True:
            hist_df["log"] = np.log(hist_df[column])
            src = ColumnDataSource(hist_df)
            plot = figure(plot_height = self.height, plot_width = self.width,
                  title = "Histogram of {}".format(column.capitalize()),
                  x_axis_label = column.capitalize(),
                  y_axis_label = "Log Count")    
            plot.quad(bottom = 0, top = "log",left = "left", 
                right = "right", source = src, fill_color = self.colors[0], 
                line_color = "black", fill_alpha = 0.7,
                hover_fill_alpha = 1.0, hover_fill_color = self.colors[1])
        else:
            src = ColumnDataSource(hist_df)
            plot = figure(plot_height = self.height, plot_width = self.width,
                  title = "Histogram of {}".format(column.capitalize()),
                  x_axis_label = column.capitalize(),
                  y_axis_label = "Count")    
            plot.quad(bottom = 0, top = column,left = "left", 
                right = "right", source = src, fill_color = self.colors[0], 
                line_color = "black", fill_alpha = 0.7,
                hover_fill_alpha = 1.0, hover_fill_color = self.colors[1])

        # hover tool
        hover = HoverTool(tooltips = [('Interval', '@interval'),
                                  ('Count', str("@" + column))])
        plot.add_tools(hover)

        # output
        if show_plot == True:
            show(plot)
        else:
            return plot

    def histotabs(self, dataframe, features, log_scale=False, show_plot=False):
        '''
        Builds tabbed interface for a series of histograms; calls hist_hover. Specifying 'show_plot=True' will simply display the histograms in sequence rather than in a tabbed interface.

        Parameters:
        ----------
        Input:
        dataframe {df}: a Pandas dataframe
        features {list -> string}: list of features to plot
        log_scale {bool}: True to plot on a log scale
        colors {list -> string}: list of colors for histogram; first color default color, second color is hover color
        show_plot {bool}: True to display the plot, False to store the plot in a variable (for use in later methods)

        Output:
        Tabbed interface for viewing interactive histograms of specified features

        '''
        hists = []
        for f in features:
            h = self.hist_hover(dataframe, f, log_scale=log_scale, show_plot=show_plot)
            p = Panel(child=h, title=f.capitalize())
            hists.append(p)
        t = Tabs(tabs=hists)
        show(t)

    def filtered_histotabs(self, dataframe, feature, filter_feature, log_scale=False, show_plot=False):
        '''
        Builds tabbed histogram interface for one feature filtered by another. Feature is numeric, fiter feature is categorical.

        Parameters:
        ----------
        Input:
        dataframe {df}: a Pandas dataframe
        features {list -> string}: list of features to plot
        log_scale {bool}: True to plot on a log scale
        colors {list -> string}: list of colors for histogram; first color default color, second color is hover color
        show_plot {bool}: True to display the plot, False to store the plot in a variable (for use in later methods)

        Output:
        Tabbed interface for viewing interactive histograms of specified feature filtered by categorical filter feature

        '''
        hists = []
        for col in dataframe[filter_feature].unique():
            sub_df = dataframe[dataframe[filter_feature] == col]
            histo = self.hist_hover(sub_df, feature, log_scale=log_scale, show_plot=show_plot)
            p = Panel(child = histo, title=col)
            hists.append(p)
        t = Tabs(tabs=hists)
        show(t)


def make_dataset(df, statuses=["viewed", "explored","certified"],
                    range_start=0, range_end=200000, bin_width=1000):
    assert range_start < range_end, "Start must be less than end!"

    by_status = pd.DataFrame(columns=['proportion', 'left', 'right',
                                     's_proportion', 's_interval',
                                     'name', 'color'])
    range_extent = range_end - range_start
    # Iterate through the statuses
    for i, status in enumerate(statuses):
        # subset by status
        subset = df[df[status]==1]
        # histogram
        st_hist, edges = np.histogram(subset["nevents"],
                        bins=int(range_extent / bin_width),
                        range = [range_start, range_end]) 
        # divide counts by total to get proportion
        st_df = pd.DataFrame({'proportion': st_hist / np.sum(st_hist),
                              'left': edges[:-1], 'right': edges[1:]})
        # format the proportion
        st_df['s_proportion'] = ['%0.5f' % proportion for proportion   in st_df['proportion']]
        # format the interval
        st_df['s_interval'] = ['%d to %d events' % (left, right) for   left, right in zip(st_df['left'], st_df['right'])]
        # assign status labels
        st_df['name'] = status
        # color for each status
        st_df['color'] = Category20_16[i]
        # add to overall dataframe
        by_status = by_status.append(st_df)
    # Overall dataframe
    by_status = by_status.sort_values(['name', 'left'])
    # Convert dataframe to column data source for bokeh
    return ColumnDataSource(by_status)

def make_plot(src, plot_title="Histogram"):
    # blank plot with correct labels
    p = figure(plot_width=700, plot_height=700, title=plot_title,
                x_axis_label="Events", y_axis_label="Count")
    
    # Quad glyphs to create a histogram
    p.quad(source=src, bottom=0, top='proportion', left='left',
          right='right', color='color', fill_alpha=0.7, hover_fill_color='color', legend='name',       hover_fill_alpha=1.0, line_color='black')
    # Hover tool with vline mode
    hover = HoverTool(tooltips=[('Status', '@name'),
                            ('Events', '@s_interval'),
                            ('Proportion', '@s_proportion')],
                            mode='vline')
    p.add_tools(hover)
    return p

def update(attr, old, new):
    # get the list of statuses for the graph
    statuses_to_plot = [status_selection.labels[i] for i in   status_selection.active]
    # make a new dataset based on selected characters
    # use make dataset function defined above
    new_src = make_dataset(statuses_to_plot,
                            range_start=0,
                            range_end=200000,
                            bin_width=1000)
    src.data.update(new_src.data)


# Bokeh app
def modify_doc(doc):

	status_selection = CheckboxGroup(labels=['viewed','explored',    'certified'], active=[0,1])
	status_selection.on_change('active', update)
	initial_status = [status_selection.labels[i] for i in     	status_selection.active]
	src = make_dataset(edX, initial_status)
	p = make_plot(src)
	controls = WidgetBox(status_selection)
	layout = row(controls, p)
	tab = Panel(child=layout, title='Events Histogram')
	tabs = Tabs(tabs=[tab])
	doc.add_root(tabs)

