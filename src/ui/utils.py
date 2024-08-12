import toml
import streamlit as st


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def write_config(config, path):
    """
    writes a config dict to the toml format
    :param config: dict of configurations
    :param path: output path
    """
    with open(path, mode="w") as out:
        return toml.dump(config, out)
