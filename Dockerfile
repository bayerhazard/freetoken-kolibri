# Thin derivative of the official FreeToken CUDA serving runtime
# (beclab/leamon2code-freetoken) that adds the Kolibri-1 model module.
# The base image already ships the full accel stack (torch cu13, flashinfer,
# sgl-kernel, FreeToken in /opt/venv), so this build is only two layers.
FROM beclab/leamon2code-freetoken:0.1.3-cu132

# The base image runs as UID 1000 and site-packages is root-owned; patch as root.
USER root
COPY kolibri /tmp/kolibri
COPY patch_register.py /tmp/patch_register.py

RUN set -eux; \
    SP="$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"; \
    cp -r /tmp/kolibri "$SP/freetoken/models/kolibri"; \
    python /tmp/patch_register.py; \
    python -c "import ast,sysconfig,os; p=os.path.join(sysconfig.get_paths()['purelib'],'freetoken/models/register.py'); ast.parse(open(p).read()); print('register.py parses')"

USER 1000:1000
