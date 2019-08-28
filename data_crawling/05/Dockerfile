FROM python:3.7.3
EXPOSE 8088
ADD . /kyrylo_kundik_hw
WORKDIR /kyrylo_kundik_hw
RUN pip3 install -e . && chmod +x ./dev.sh && chmod +x ./init_migrations.sh
#RUN init_migrations.sh
CMD ["./dev.sh"]
