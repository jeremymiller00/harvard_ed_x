import numpy as np 
import pandas as pd
import seaborn as sns
sns.reset_defaults
sns.set_style(style='darkgrid')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
font = {'size'   : 16}
plt.rc('font', **font)
plt.ion()
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams["figure.figsize"] = (20.0, 10.0)
pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 2000)
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Panel, CategoricalColorMapper
from bokeh.models.widgets import CheckboxGroup, Tabs, Panel
from bokeh.layouts import column, row, WidgetBox
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.palettes import Category20_16
from bokeh.layouts import row, column, gridplot
import bokeh_histogram

get_ipython().run_line_magic('matplotlib', 'inline')
output_notebook()

# Plotting Year of Birth against numerical columns

def yob_scatter(df, y_feature):
    '''
    Plot numeric features against year of birth
    '''

    p = figure(x_axis_label="Year of Birth", y_axis_label=y_feature, 
               title='Year of Birth vs {}'.format(y_feature),
               tools='pan, box_zoom, reset')
    h = HoverTool(tooltips=None, mode='vline')
    p.circle(df['YoB'], df[y_feature], alpha=0.3, hover_color="red")
    p.add_tools(h)
    show(p)


'''
p = figure(x_axis_label="Year of Birth", y_axis_label="Events", 
           title='Year of Birth vs Number of Events',
           tools='pan, box_zoom, reset')
h = HoverTool(tooltips=None, mode='vline')
p.circle(df_small['YoB'], df_small['nevents'], alpha=0.3, hover_color="red")
p.add_tools(h)
show(p)

p = figure(x_axis_label="Year of Birth", y_axis_label="Days Accessed", 
           title='Year of Birth vs Number of Days Accessed',
           tools='pan, box_zoom, reset')
p.circle(df_small['YoB'], df_small['ndays_act'], alpha=0.3)
show(p)

p = figure(x_axis_label="Year of Birth", y_axis_label="Video Plays", 
           title='Year of Birth vs Number of Video Plays',
           tools='pan, box_zoom, reset')
p.circle(df_small['YoB'], df_small['nplay_video'], alpha=0.3)
show(p)

p = figure(x_axis_label="Year of Birth", y_axis_label="Chapters Accessed", 
           title='Year of Birth vs Number of Chapters Accessed',
           tools='pan, box_zoom, reset')
p.circle(df_small['YoB'], df_small['nchapters'], alpha=0.3)
show(p)

p = figure(x_axis_label="Year of Birth", y_axis_label="Forum Posts", 
           title='Year of Birth vs Number of Forum Posts',
           tools='pan, box_zoom, reset')
p.circle(df_small['YoB'], df_small['nforum_posts'], size=6, alpha=0.3)
show(p)


# Year of Birth Against Numerical Columns, colored by gender
src = ColumnDataSource(df_small)

p = figure(x_axis_label="Year of Birth", y_axis_label="Events", 
           title='Year of Birth vs Number of Events',
           tools='pan, box_zoom, box_select, lasso_select, reset')
m = CategoricalColorMapper(factors=['m', 'f', 'NaN'], 
                           palette=['green', 'blue', 'red'])
h = HoverTool(tooltips=None, mode='hline')
p.circle('YoB', 'nevents', source=src, alpha=0.3, 
         color={'field':'gender', 'transform':m}, legend='gender',
        hover_color='red')
p.add_tools(h)
show(p)

p = figure(x_axis_label="Year of Birth", y_axis_label="Days Accessed", 
           title='Year of Birth vs Number of Days Accessed',
           tools='pan, box_zoom, reset')
m = CategoricalColorMapper(factors=['m', 'f', 'NaN'], 
                           palette=['green', 'blue', 'red'])
p.circle('YoB', 'ndays_act', source=src, alpha=0.3, 
         color={'field':'gender', 'transform':m}, legend='gender')
show(p)

p = figure(x_axis_label="Year of Birth", y_axis_label="Video Plays", 
           title='Year of Birth vs Number of Video Plays',
           tools='pan, box_zoom, reset')
m = CategoricalColorMapper(factors=['m', 'f', 'NaN'], 
                           palette=['green', 'blue', 'red'])
p.circle('YoB', 'nplay_video', source=src, alpha=0.3, 
         color={'field':'gender', 'transform':m}, legend='gender')
show(p)

p = figure(x_axis_label="Year of Birth", y_axis_label="Chapters Viewed", 
           title='Year of Birth vs Number of Chapters Viewed',
           tools='pan, box_zoom, reset')
m = CategoricalColorMapper(factors=['m', 'f', 'NaN'], 
                           palette=['green', 'blue', 'red'])
p.circle('YoB', 'nchapters', source=src, alpha=0.2, 
         color={'field':'gender', 'transform':m}, legend='gender')
show(p)

p = figure(x_axis_label="Year of Birth", y_axis_label="Forum Posts", 
           title='Year of Birth vs Number of Forum Posts',
           tools='pan, box_zoom, reset')
m = CategoricalColorMapper(factors=['m', 'f', 'NaN'], 
                           palette=['green', 'blue', 'red'])
p.circle('YoB', 'nforum_posts', source=src, alpha=0.3, 
         color={'field':'gender', 'transform':m}, legend='gender')
show(p)

# Year of Birth Against Numerical Columns, colored by completion

p = figure(x_axis_label="Year of Birth", y_axis_label="Events", 
           title='Year of Birth vs Number of Events (Colored by Completion)',
           tools='pan, box_zoom, reset')
m = CategoricalColorMapper(factors=['n', 'y'], 
                           palette=['orange', 'green'])
p.circle('YoB', 'nevents', source=src, alpha=0.3, 
         color={'field':'completed', 'transform':m}, legend='completed')
show(p)

sp = figure(x_axis_label="Year of Birth", y_axis_label="Video Plays", 
           title='Year of Birth vs Number of Video Plays (Colored by Completion)',
           tools='pan, box_zoom, reset')
m = CategoricalColorMapper(factors=['n', 'y'], 
                           palette=['orange', 'green'])
p.circle('YoB', 'nplay_video', source=src, alpha=0.3, 
         color={'field':'completed', 'transform':m}, legend='completed')
show(p)

p = figure(x_axis_label="Year of Birth", y_axis_label="Chapters Viewed", 
           title='Year of Birth vs Chapters Viewed (Colored by Completion)',
           tools='pan, box_zoom, reset')
m = CategoricalColorMapper(factors=['n', 'y'], 
                           palette=['orange', 'green'])
p.circle('YoB', 'nchapters', source=src, alpha=0.3,
         color={'field':'completed', 'transform':m}, legend='completed')
show(p)


m = CategoricalColorMapper(factors=['n', 'y'], 
                           palette=['orange', 'green'])

p1 = figure(x_axis_label="Year of Birth", y_axis_label="Events", 
           title='Year of Birth vs Number of Events (Colored by Completion)',
           tools='pan, box_zoom, box_select, lasso_select, reset')
p1.circle('YoB', 'nevents', source=src, alpha=0.3, 
         color={'field':'completed', 'transform':m}, legend='completed')

p2 = figure(x_axis_label="Year of Birth", y_axis_label="Video Plays", 
           title='Year of Birth vs Number of Video Plays (Colored by Completion)',
           tools='pan, box_zoom, box_select, lasso_select, reset')
p2.circle('YoB', 'nplay_video', source=src, alpha=0.3, 
         color={'field':'completed', 'transform':m}, legend='completed')

p3 = figure(x_axis_label="Year of Birth", y_axis_label="Chapters Viewed", 
           title='Year of Birth vs Chapters Viewed (Colored by Completion)',
           tools='pan, box_zoom, box_select, lasso_select, reset')
p3.circle('YoB', 'nchapters', source=src, alpha=0.3,
         color={'field':'completed', 'transform':m}, legend='completed')

l = row(p1, p2, p3)
show(l)


m = CategoricalColorMapper(factors=['n', 'y'], 
                           palette=['orange', 'green'])

p1 = figure(x_axis_label="Year of Birth", y_axis_label="Events", 
           title='Year of Birth vs Number of Events (Colored by Completion)',
           tools='pan, box_zoom, reset')
p1.circle('YoB', 'nevents', source=src, alpha=0.3, 
         color={'field':'completed', 'transform':m}, legend='completed')

p2 = figure(x_axis_label="Year of Birth", y_axis_label="Video Plays", 
           title='Year of Birth vs Number of Video Plays (Colored by Completion)',
           tools='pan, box_zoom, reset')
p2.circle('YoB', 'nplay_video', source=src, alpha=0.3, 
         color={'field':'completed', 'transform':m}, legend='completed')

p3 = figure(x_axis_label="Year of Birth", y_axis_label="Chapters Viewed", 
           title='Year of Birth vs Chapters Viewed (Colored by Completion)',
           tools='pan, box_zoom, reset')
p3.circle('YoB', 'nchapters', source=src, alpha=0.3,
         color={'field':'completed', 'transform':m}, legend='completed')

first = Panel(child=p1, title="Events")
second = Panel(child=p2, title="Video Plays")
third = Panel(child=p3, title="Chapters Viewed")
t = Tabs(tabs=[first, second, third])
show(t)

filtered_histotabs(df, 'nevents', 'final_cc_cname_DI', log_scale=True)

#hover histograms of events with each country as a tab
hists = []
for c in df['final_cc_cname_DI'].unique():
    sub_df = df[df['final_cc_cname_DI'] == c]
    h = hist_hover(sub_df, 'nevents', colors=["SteelBlue", "Tan"], log_scale=True, show_plot=False)
    p = Panel(child=h, title=c)
    hists.append(p)
t = Tabs(tabs=hists)
show(t)

src = ColumnDataSource(df)

m = CategoricalColorMapper(factors=['n', 'y'], 
                           palette=['orange', 'green'])

p1 = figure(x_axis_label="Year of Birth", y_axis_label="Events", 
           title='Year of Birth vs Number of Events (Colored by Completion)',
           tools='pan, box_zoom, reset')
p1.circle('YoB', 'nevents', source=src, alpha=0.3, 
         color={'field':'completed', 'transform':m}, legend='completed')

p2 = figure(x_axis_label="Year of Birth", y_axis_label="Video Plays", 
           title='Year of Birth vs Number of Video Plays (Colored by Completion)',
           tools='pan, box_zoom, reset')
p2.circle('YoB', 'nplay_video', source=src, alpha=0.3, 
         color={'field':'completed', 'transform':m}, legend='completed')

p3 = figure(x_axis_label="Year of Birth", y_axis_label="Chapters Viewed", 
           title='Year of Birth vs Chapters Viewed (Colored by Completion)',
           tools='pan, box_zoom, reset')
p3.circle('YoB', 'nchapters', source=src, alpha=0.3,
         color={'field':'completed', 'transform':m}, legend='completed')

l = row(p1, p2, p3)
show(l)

hist_hover(df,"nevents",colors=["orange", "yellow"],log_scale=True)

hist_hover(df,"ndays_act",colors=["orange", "yellow"],log_scale=True)

hist_hover(df.fillna(value=0),"nchapters",colors=["orange", "yellow"],log_scale=True)

hist_hover(df.fillna(value=0),"nplay_video",colors=["orange", "yellow"],log_scale=True)

hist_hover(df.fillna(value=0),"nforum_posts",colors=["orange", "yellow"],log_scale=True)
'''

################################################################
if __name__ == "__main__":

    df = pd.read_csv('data/cleaned_harvard.csv')

    h = BokehHistogram()

    h.hist_hover(df.fillna(value=0, axis=1), 'nchapters', log_scale=True)

    h.histotabs(df.fillna(value=0, axis=1), 
                features = ['nevents', 'ndays_act', 'nplay_video', 'nchapters', 
                            'nforum_posts'],
                log_scale=True)

    h.filtered_histotabs(df.fillna(value=0, axis=1), 'nevents', 'final_cc_cname_DI', 
                log_scale=True)
