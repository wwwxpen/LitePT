FROM imotion-cn-beijing.cr.volces.com/imotion-img-space/occ_litept:20260323_installnumba

# 设置环境变量（在 SHELL 之前设置）
ENV PATH="/opt/conda/envs/base/bin:/opt/conda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# 复制应用代码
RUN rm -rf /root/LitePT
RUN mkdir -p /root/LitePT
COPY configs /root/LitePT/configs
COPY da_binds /root/LitePT/da_binds
COPY datasets /root/LitePT/datasets
COPY deploy /root/LitePT/deploy
COPY engines /root/LitePT/engines
COPY libs /root/LitePT/libs
COPY litept /root/LitePT/litept
COPY metrics /root/LitePT/metrics
COPY models /root/LitePT/models
COPY scripts /root/LitePT/scripts
COPY tools /root/LitePT/tools
COPY utils /root/LitePT/utils
COPY *.py /root/LitePT/
COPY *.sh /root/LitePT/

# 设置工作目录
WORKDIR /root/LitePT

# 激活conda环境并编译da_binds
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# 安装 Python 包（在复制代码之前安装可以更好地利用 Docker 缓存）
# RUN conda run -n base pip install --no-cache-dir numba

# 设置默认命令（可选）
CMD ["/bin/bash"]