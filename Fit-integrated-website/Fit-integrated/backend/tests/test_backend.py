import importlib.util
from pathlib import Path
import threading
import unittest
import urllib.request
import urllib.error
import json
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
server=load('fitserver',ROOT/'run_fit.py')
runner=load('scanner_runner',ROOT/'backend/scanner_runner.py')
class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        cls.thread=threading.Thread(target=cls.http.serve_forever,daemon=True);cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.http.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown();cls.http.server_close();cls.thread.join()
    def call(self,path,data=None,token=None,origin=None,host=None):
        headers={}
        if token:headers['X-Fit-Session']=token
        if origin:headers['Origin']=origin
        if host:headers['Host']=host
        req=urllib.request.Request(self.base+path,data=json.dumps(data).encode() if data is not None else None,headers=headers)
        try:
            with urllib.request.urlopen(req) as r:return r.status,json.load(r)
        except urllib.error.HTTPError as e:return e.code,json.load(e)
    def test_health(self):
        status,data=self.call('/api/health');self.assertEqual(status,200);self.assertTrue(data['ok']);self.assertIn('sessionToken',data)
    def test_session_required(self):
        self.assertEqual(self.call('/api/scan')[0],403)
        self.assertEqual(self.call('/api/scan',{'heightCm':165})[0],403)
    def test_cross_origin_cannot_launch_camera(self):
        self.assertEqual(self.call('/api/scan',{'heightCm':165},server.TOKEN,'https://other.example')[0],403)
    def test_untrusted_host(self):
        self.assertEqual(self.call('/api/health',host='other.example')[0],403)
    def test_height_validation(self):
        self.assertEqual(self.call('/api/scan',{'heightCm':0},server.TOKEN)[0],400)
    def test_delete_clears_profile(self):
        server.JOB={'status':'complete','profile':{'private':'test'}}
        self.assertEqual(self.call('/api/scan/cancel',{},server.TOKEN)[0],200)
        self.assertEqual(self.call('/api/scan',token=server.TOKEN)[1],{'status':'cancelled'})
    def test_original_python_rows_map_to_browser_profile(self):
        rows=[('Bust / chest',104,2,''),('Waist',82,2,''),('Hips',112,2,''),('Inseam',None,None,'')]
        profile=runner.normalize_rows(rows,165,True)
        self.assertEqual(profile['source'],'python');self.assertEqual(profile['measurementsCm']['bust'],104)
        self.assertNotIn('inseam',profile['measurementsCm'])
        with self.assertRaises(ValueError):runner.normalize_rows([],165,True)

if __name__=='__main__':unittest.main()
