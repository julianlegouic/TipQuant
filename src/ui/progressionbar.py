from time import time

import streamlit as st


class ProgressionBar:
    """Progression Bar allow display progression bar with status and remaining time"""
    def __init__(self):
        self.status = st.empty()
        self.sub_status = st.empty()
        self.prog_bar = st.progress(0)
        self.iteration = 0
        self.length = None
        self.last_iter_time = None
        self.avg_time = None

    def __call__(self, status, length):
        """
        Initializes the progress bar and status
        :param status: status text
        :type status: str
        :param length: Number of iterations
        :type length: int
        """
        self.status.info(status)
        self.prog_bar.progress(0)
        self.length = length
        self.iteration = 0
        self.last_iter_time = time()

    def update(self, alpha=0.9):
        """
        Updates the progress bar and the sub status with by current iteration number
        and left time
        :param alpha: Remaining time smoothing parameter
        :type alpha: float
        """
        self.iteration += 1
        current_prog = self.iteration / self.length
        self.prog_bar.progress(current_prog)
        current_time = time()
        if self.avg_time is not None:
            self.avg_time = alpha*self.avg_time + (1-alpha)*(current_time-self.last_iter_time)
            time_left = (self.length-self.iteration) * self.avg_time
            self.sub_status.text(f"Step {self.iteration}/{self.length}, time left :{time_left:.0f}s")
        else:
            self.avg_time = current_time - self.last_iter_time
            self.sub_status.text(f"Step {self.iteration}/{self.length}")
        self.last_iter_time = current_time

    def success(self, status):
        """
        Updates the status and hide the progression bar and sub status
        :param status: success status text
        :type status: str
        """
        self.status.success(status)
        self.sub_status.empty()
        self.prog_bar.empty()
