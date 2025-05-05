console.log("clientside.js loaded!");

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.clientside = window.dash_clientside.clientside || {};

window.dash_clientside.clientside.download_plot = function(n_clicks) {
    if (!n_clicks) return window.dash_clientside.no_update;

    const container = document.getElementById('scatter-plot');
    const plot = container && container.querySelector('.js-plotly-plot');

    if (!plot) {
        console.log("Plot element not found.");
        return window.dash_clientside.no_update;
    }

    // Try to extract the plot title from layout
    const title = plot.layout && plot.layout.title && plot.layout.title.text ? plot.layout.title.text : 'code_similarity_plot';

    // Sanitize title for use in a filename
    const safeTitle = title.replace(/[^\w\-_. ]/g, '_');

    Plotly.toImage(plot, {
        format: 'png',
        width: 1240,
        height: 1754
    }).then(function(dataUrl) {
        const a = document.createElement('a');
        a.href = dataUrl;
        a.download = `${safeTitle}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });

    return 0;
};
