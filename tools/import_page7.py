import importlib.util, traceback
p='page/07_conclusion_critique_perspective.py'
try:
    spec = importlib.util.spec_from_file_location('page7', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print('IMPORT_OK')
    if hasattr(mod, 'run'):
        try:
            mod.run()
            print('RUN_OK')
        except Exception as e:
            print('RUN_ERR')
            traceback.print_exc()
    else:
        print('NO_RUN')
except Exception:
    print('IMPORT_ERR')
    traceback.print_exc()
