FROM python:3.7.3
WORKDIR ./
COPY requirements/requirements.dev.txt ./
RUN pip install --trusted-host pypi.python.org -r requirements.dev.txt
COPY ./ ./
EXPOSE 8080
CMD ["sh", "dev.sh"]