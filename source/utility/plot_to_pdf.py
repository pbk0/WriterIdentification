"""
Utility to plot figures in pdf and latex so that they can be used directly in final report
http://matplotlib.org/users/style_sheets.html#style-sheets
http://matplotlib.org/faq/howto_faq.html
http://matplotlib.org/users/usetex.html
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib


if __name__ == '__main__':
    df = pd.DataFrame(np.random.randn(10,2))
    df.columns = ['Column 1', 'Column 2']
    ax = df.plot()
    ax.set_xlabel("X label")
    ax.set_ylabel("Y label")
    ax.set_title("Title")
    plt.tight_layout()
    plt.savefig("image1.pdf")
