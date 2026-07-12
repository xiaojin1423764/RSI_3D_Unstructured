function out = interpolate_to_structured_grid(csvFile, outFile, gridN, bounds)
%INTERPOLATE_TO_STRUCTURED_GRID Interpolate cell-centered CSV data to a Cartesian grid.
%
% Usage:
%   addpath('examples');
%   interpolate_to_structured_grid();
%   interpolate_to_structured_grid('examples/csv_data/Rec/figure5_RSI.csv', ...
%       'examples/matlab_grid/Rec_figure5_RSI_grid.mat', 101, [0 1; 0 1; 0 1]);
%
% The input CSV must contain x, y, z, and phi0 columns. The output MAT file
% stores xq, yq, zq, X, Y, Z, phi0Grid, and metadata.

    thisFile = mfilename('fullpath');
    examplesDir = fileparts(thisFile);

    if nargin < 1 || isempty(csvFile)
        csvFile = fullfile(examplesDir, 'csv_data', 'Rec', 'figure5_RSI.csv');
    end
    if nargin < 3 || isempty(gridN)
        gridN = 101;
    end
    if isscalar(gridN)
        gridN = [gridN, gridN, gridN];
    end
    if numel(gridN) ~= 3
        error('gridN must be a scalar or a three-element vector.');
    end

    T = readtable(csvFile);
    required = {'x', 'y', 'z', 'phi0'};
    for k = 1:numel(required)
        if ~ismember(required{k}, T.Properties.VariableNames)
            error('Input CSV is missing column: %s', required{k});
        end
    end

    x = T.x;
    y = T.y;
    z = T.z;
    phi0 = T.phi0;

    if nargin < 4 || isempty(bounds)
        bounds = [
            min(x), max(x);
            min(y), max(y);
            min(z), max(z)
        ];
    end
    if ~isequal(size(bounds), [3, 2])
        error('bounds must be a 3-by-2 matrix: [xmin xmax; ymin ymax; zmin zmax].');
    end

    xq = linspace(bounds(1, 1), bounds(1, 2), gridN(1));
    yq = linspace(bounds(2, 1), bounds(2, 2), gridN(2));
    zq = linspace(bounds(3, 1), bounds(3, 2), gridN(3));
    [X, Y, Z] = ndgrid(xq, yq, zq);

    F = scatteredInterpolant(x, y, z, phi0, 'linear', 'nearest');
    phi0Grid = F(X, Y, Z);

    if nargin < 2 || isempty(outFile)
        [csvDir, csvBase, ~] = fileparts(csvFile);
        [~, caseName] = fileparts(csvDir);
        if isempty(caseName) || strcmp(caseName, 'csv_data')
            caseName = 'field';
        end
        outDir = fullfile(examplesDir, 'matlab_grid');
        outFile = fullfile(outDir, sprintf('%s_%s_grid.mat', caseName, csvBase));
    end
    outDir = fileparts(outFile);
    if ~isempty(outDir) && ~isfolder(outDir)
        mkdir(outDir);
    end

    metadata = struct();
    metadata.csvFile = csvFile;
    metadata.gridN = gridN;
    metadata.bounds = bounds;
    metadata.method = 'scatteredInterpolant linear, nearest extrapolation';
    metadata.valueName = 'phi0';

    save(outFile, 'xq', 'yq', 'zq', 'X', 'Y', 'Z', 'phi0Grid', 'metadata', '-v7.3');

    [outDir, outBase, ~] = fileparts(outFile);
    midY = max(1, round(numel(yq) / 2));
    slicePng = fullfile(outDir, [outBase, '_y_mid.png']);
    sliceCsv = fullfile(outDir, [outBase, '_y_mid.csv']);

    write_mid_y_slice_csv(sliceCsv, xq, zq, squeeze(phi0Grid(:, midY, :)));
    write_mid_y_slice_png(slicePng, xq, zq, squeeze(phi0Grid(:, midY, :)), yq(midY));

    out = struct();
    out.matFile = outFile;
    out.slicePng = slicePng;
    out.sliceCsv = sliceCsv;
    out.gridN = gridN;
    out.bounds = bounds;
end


function write_mid_y_slice_csv(fileName, xq, zq, values)
    fid = fopen(fileName, 'w');
    if fid < 0
        error('Cannot open output file: %s', fileName);
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, 'x,z,phi0\n');
    for i = 1:numel(xq)
        for k = 1:numel(zq)
            fprintf(fid, '%.16g,%.16g,%.16g\n', xq(i), zq(k), values(i, k));
        end
    end
end


function write_mid_y_slice_png(fileName, xq, zq, values, yValue)
    fig = figure('Visible', 'off', 'Color', 'w');
    imagesc(xq, zq, values.');
    set(gca, 'YDir', 'normal');
    axis image tight;
    xlabel('x');
    ylabel('z');
    title(sprintf('Interpolated phi0, y = %.4g', yValue));
    if exist('turbo', 'file') == 2
        colormap(turbo);
    else
        colormap(parula);
    end
    colorbar;
    if exist('exportgraphics', 'file') == 2
        exportgraphics(fig, fileName, 'Resolution', 300);
    else
        print(fig, fileName, '-dpng', '-r300');
    end
    close(fig);
end
