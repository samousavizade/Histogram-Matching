import cv2 as cv
import numpy as np
import matplotlib.pyplot as pp
from functools import reduce
from skimage.exposure import match_histograms as builtin_mh


def size_of(image):
    return reduce(lambda a, b: a * b, image.shape)


def histogram_matching_transformation(src_cdf, ref_cdf):
    transformation = np.zeros(256)
    found_val = 0
    len_cdf = len(src_cdf)
    for src_idx in range(len_cdf):
        for ref_idx in range(len_cdf):
            if ref_cdf[ref_idx] >= src_cdf[src_idx]:
                found_val = ref_idx
                break

        transformation[src_idx] = found_val

    return transformation


def match_histograms(src_channel, ref_channel):
    # get histogram
    src_histogram, src_bins = np.histogram(src_channel.flatten(), 256, (0, 256))
    ref_histogram, ref_bins = np.histogram(ref_channel.flatten(), 256, (0, 256))

    # calculate src, ref pdf
    src_pdf = src_histogram / size_of(src_channel)
    ref_pdf = ref_histogram / size_of(ref_channel)

    # calculate src, ref cdf
    src_cdf = src_pdf.cumsum()
    ref_cdf = ref_pdf.cumsum()

    # get transformation
    transformation = histogram_matching_transformation(src_cdf, ref_cdf)

    # use transformation on src_channel
    result = transformation[src_channel]

    return result


def main():
    source_file_name = 'input/source.jpg'
    reference_file_name = 'input/reference.jpg'

    output_file_name = 'output/output.jpg'
    histograms_file_name = 'output/histograms.jpg'

    src = cv.imread(source_file_name, cv.IMREAD_COLOR)
    ref = cv.imread(reference_file_name, cv.IMREAD_COLOR)

    # src -> out based on ref and check with built_in histogram_matching
    out = np.zeros(src.shape, dtype='uint8')

    for i in range(3):
        out[:, :, i] = match_histograms(src[:, :, i], ref[:, :, i])
        # check with build in histogram matching function
        # out[:, :, i] = builtin_mh(src[:, :, i], ref[:, :, i])

    # plot results
    fig, out = plot_result(out, ref, src)

    pp.imsave(output_file_name, out)

    fig.savefig(histograms_file_name)
    fig.show()
    # while not fig.waitforbuttonpress(): pass


def plot_result(out, ref, src):
    fig, ((r_pdf_ax, g_pdf_ax, b_pdf_ax, src_ax), (r_cdf_ax, g_cdf_ax, b_cdf_ax, out_ax)) = pp.subplots(nrows=2,
                                                                                                        ncols=4)
    fig: pp.Figure
    fig.set_size_inches(16, 8)
    ref_hist, bins = np.histogram(ref[:, :, 0].flatten(), bins=256, range=(0, 256))
    ref_pdf = ref_hist / size_of(ref[:, :, 0])
    r_pdf_ax.bar(bins[:-1], ref_pdf, fc=(1, 0, 0, .4), label='Reference PDF')
    out_hist, bins = np.histogram(out[:, :, 0].flatten(), bins=256, range=(0, 256))
    out_pdf = out_hist / size_of(out[:, :, 0])
    r_pdf_ax.bar(bins[:-1], out_pdf, fc=(0, 0, 1, .4), label='Output PDF')
    r_pdf_ax.set_title('Red Channel PDFs')
    r_pdf_ax.legend()
    ref_cdf = np.cumsum(ref_pdf)
    r_cdf_ax.bar(bins[:-1], ref_cdf, fc=(0, 1, 0, .4), label='Reference CDF')
    out_cdf = np.cumsum(out_pdf)
    r_cdf_ax.bar(bins[:-1], out_cdf, fc=(0, 0, 1, .4), label='Output CDF')
    r_cdf_ax.set_title('Red Channel CDFs')
    r_cdf_ax.legend()
    ref_hist, bins = np.histogram(ref[:, :, 1].flatten(), bins=256, range=(0, 256))
    ref_pdf = ref_hist / size_of(ref[:, :, 1])
    g_pdf_ax.bar(bins[:-1], ref_pdf, fc=(1, 0, 0, .4), label='Reference PDF')
    out_hist, bins = np.histogram(out[:, :, 1].flatten(), bins=256, range=(0, 256))
    out_pdf = out_hist / size_of(out[:, :, 1])
    g_pdf_ax.bar(bins[:-1], out_pdf, fc=(0, 0, 1, .4), label='Output PDF')
    g_pdf_ax.set_title('Green Channel PDFs')
    g_pdf_ax.legend()
    ref_cdf = np.cumsum(ref_pdf)
    g_cdf_ax.bar(bins[:-1], ref_cdf, fc=(0, 1, 0, .4), label='Reference CDF')
    out_cdf = np.cumsum(out_pdf)
    g_cdf_ax.bar(bins[:-1], out_cdf, fc=(0, 0, 1, .4), label='Output CDF')
    g_cdf_ax.set_title('Green Channel CDFs')
    g_cdf_ax.legend()
    ref_hist, bins = np.histogram(ref[:, :, 2].flatten(), bins=256, range=(0, 256))
    ref_pdf = ref_hist / size_of(ref[:, :, 2])
    b_pdf_ax.bar(bins[:-1], ref_pdf, fc=(1, 0, 0, .4), label='Reference PDF')
    out_hist, bins = np.histogram(out[:, :, 2].flatten(), bins=256, range=(0, 256))
    out_pdf = out_hist / size_of(out[:, :, 2])
    b_pdf_ax.bar(bins[:-1], out_pdf, fc=(0, 0, 1, .4), label='Output PDF')
    b_pdf_ax.set_title('Blue Channel PDFs')
    b_pdf_ax.legend()
    ref_cdf = np.cumsum(ref_pdf)
    b_cdf_ax.bar(bins[:-1], ref_cdf, fc=(0, 1, 0, .4), label='Reference CDF')
    out_cdf = np.cumsum(out_pdf)
    b_cdf_ax.bar(bins[:-1], out_cdf, fc=(0, 0, 1, .4), label='Output CDF')
    b_cdf_ax.set_title('Blue Channel CDFs')
    b_cdf_ax.legend()
    src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    src_ax.imshow(src, vmin=0, vmax=255)
    src_ax.set_title('Source')
    out = cv.cvtColor(out, cv.COLOR_BGR2RGB)
    out_ax.imshow(out, vmin=0, vmax=255)
    out_ax.set_title('Output')
    return fig, out


if __name__ == '__main__':
    main()
