from database import get_connection, process_pipeline_item


class ApartmentsPipeline(object):

    def open_spider(self, spider):
        self.conn, self.cur = get_connection(database='apartments')

    def close_spider(self, spider):
        self.cur.close()
        self.conn.close()

    def process_item(self, item, spider):
        status = process_pipeline_item(self.conn, item)
        if status == 'process ok':
            self.conn.commit()
        elif status == 'psycopg error':
            self.cur.close()
            self.conn.close()
            self.conn, self.cur = get_connection(database='apartments')
        return item
