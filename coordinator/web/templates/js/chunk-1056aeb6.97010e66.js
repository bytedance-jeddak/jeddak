(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-1056aeb6"],{"0cb2":function(e,t,a){var r=a("7b0b"),o=Math.floor,n="".replace,l=/\$([$&'`]|\d{1,2}|<[^>]*>)/g,s=/\$([$&'`]|\d{1,2})/g;e.exports=function(e,t,a,i,d,c){var u=a+e.length,p=i.length,f=s;return void 0!==d&&(d=r(d),f=l),n.call(c,f,(function(r,n){var l;switch(n.charAt(0)){case"$":return"$";case"&":return e;case"`":return t.slice(0,a);case"'":return t.slice(u);case"<":l=d[n.slice(1,-1)];break;default:var s=+n;if(0===s)return r;if(s>p){var c=o(s/10);return 0===c?r:c<=p?void 0===i[c-1]?n.charAt(1):i[c-1]+n.charAt(1):r}l=i[s-1]}return void 0===l?"":l}))}},"2b29":function(e,t,a){"use strict";a("ee99")},5319:function(e,t,a){"use strict";var r=a("d784"),o=a("d039"),n=a("825a"),l=a("a691"),s=a("50c4"),i=a("577e"),d=a("1d80"),c=a("8aa5"),u=a("0cb2"),p=a("14c3"),f=a("b622"),h=f("replace"),m=Math.max,g=Math.min,_=function(e){return void 0===e?e:String(e)},b=function(){return"$0"==="a".replace(/./,"$0")}(),v=function(){return!!/./[h]&&""===/./[h]("a","$0")}(),k=!o((function(){var e=/./;return e.exec=function(){var e=[];return e.groups={a:"7"},e},"7"!=="".replace(e,"$<a>")}));r("replace",(function(e,t,a){var r=v?"$":"$0";return[function(e,a){var r=d(this),o=void 0==e?void 0:e[h];return void 0!==o?o.call(e,r,a):t.call(i(r),e,a)},function(e,o){var d=n(this),f=i(e);if("string"===typeof o&&-1===o.indexOf(r)&&-1===o.indexOf("$<")){var h=a(t,d,f,o);if(h.done)return h.value}var b="function"===typeof o;b||(o=i(o));var v=d.global;if(v){var k=d.unicode;d.lastIndex=0}var x=[];while(1){var w=p(d,f);if(null===w)break;if(x.push(w),!v)break;var y=i(w[0]);""===y&&(d.lastIndex=c(f,s(d.lastIndex),k))}for(var $="",I=0,D=0;D<x.length;D++){w=x[D];for(var M=i(w[0]),j=m(g(l(w.index),f.length),0),A=[],C=1;C<w.length;C++)A.push(_(w[C]));var R=w.groups;if(b){var S=[M].concat(A,j,f);void 0!==R&&S.push(R);var L=i(o.apply(void 0,S))}else L=u(M,f,j,A,R,o);j>=I&&($+=f.slice(I,j)+L,I=j+M.length)}return $+f.slice(I)}]}),!k||!b||v)},b679:function(e,t,a){"use strict";a.r(t);var r=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"model"},[a("el-card",{staticClass:"box-card"},[a("el-table",{staticStyle:{width:"100%"},attrs:{data:e.model_data}},[a("el-table-column",{attrs:{prop:"model_id",label:"模型ID"}}),a("el-table-column",{attrs:{prop:"create_time",label:"模型创建时间"}}),a("el-table-column",{attrs:{label:"备注"},scopedSlots:e._u([{key:"default",fn:function(t){return[a("el-input",{attrs:{type:"textarea",autosize:"",placeholder:"暂无"},on:{change:function(a){return e.changeAlias(t.row)}},model:{value:t.row.alias,callback:function(a){e.$set(t.row,"alias",a)},expression:"scope.row.alias"}})]}}])}),a("el-table-column",{attrs:{label:"操作"},scopedSlots:e._u([{key:"default",fn:function(t){return[a("el-popconfirm",{staticClass:"el-popconfirm oper-btn",attrs:{title:"确定要删除模型吗？"},on:{confirm:function(a){return e.deleteModel(t.row.model_id)}}},[a("el-button",{attrs:{slot:"reference",type:"danger",size:"small"},slot:"reference"},[e._v(" 删除模型 ")])],1),a("el-button",{staticClass:"oper-btn",attrs:{type:"primary",size:"small"},on:{click:function(a){return e.$router.push("/task_detail?task_id="+t.row.model_id)}}},[e._v(" 查看任务 ")]),a("el-button",{ref:e.buttonRefAndID("loadAndUnload",t.$index),staticClass:"oper-btn",attrs:{type:"warning",id:e.buttonRefAndID("loadAndUnload",t.$index),size:"small"},on:{click:function(a){return e.clickLoadAndUnload(t)}}},[e._v(" 加载模型 ")]),a("el-button",{ref:e.buttonRefAndID("predict",t.$index),staticClass:"oper-btn",attrs:{type:"info",id:e.buttonRefAndID("predict",t.$index),size:"small"},on:{click:function(a){return e.clickPredictModel(t)}}},[e._v(" 模型预测 ")])]}}])})],1),a("el-pagination",{attrs:{id:"pagination",background:"","page-size":e.page_size,"page-count":e.page_count,total:e.total_count,layout:"prev, pager, next, jumper"},on:{"current-change":e.handleCurrentChange}})],1),a("el-dialog",{attrs:{title:"模型预测",visible:e.predictModelVisible,"close-on-click-modal":!1,width:"30%"},on:{"update:visible":function(t){e.predictModelVisible=t}}},[a("el-form",{ref:"form",attrs:{model:e.predictDataForm,"label-width":"80px"}},[a("el-form-item",{attrs:{label:"属性/特征"}},[a("el-input",{attrs:{type:"textarea",placeholder:"输入预测数据的id及特征，用逗号分割，如：id,x0,x1,x2,x3,x4"},model:{value:e.predictDataForm.headers,callback:function(t){e.$set(e.predictDataForm,"headers",t)},expression:"predictDataForm.headers"}})],1),a("el-form-item",{attrs:{label:"数据"}},[a("el-input",{attrs:{type:"textarea",rows:6,placeholder:"输入要预测的数据，用逗号分割，行之间用分号分割，或直接换行。\n如：133,0.259,-1.033,0.256,0.014,-0.366;134,0.359,-1.233,0.226,0.414,-0.356\n或者：\n133,0.259,-1.033,0.256,0.014,-0.366\n134,0.359,-1.233,0.226,0.414,-0.356"},model:{value:e.predictDataForm.data,callback:function(t){e.$set(e.predictDataForm,"data",t)},expression:"predictDataForm.data"}})],1),a("el-form-item",{attrs:{label:"预测结果"}},[a("el-input",{attrs:{type:"textarea",rows:5,placeholder:"在这里显示预测结果"},model:{value:e.predictDataForm.result,callback:function(t){e.$set(e.predictDataForm,"result",t)},expression:"predictDataForm.result"}})],1)],1),a("span",{staticClass:"dialog-footer",attrs:{slot:"footer"},slot:"footer"},[a("el-button",{attrs:{type:"primary"},on:{click:e.startPredictButton}},[e._v("预 测")]),a("el-button",{on:{click:function(t){e.predictModelVisible=!1}}},[e._v("取 消")])],1)],1)],1)},o=[],n=a("1da1"),l=(a("96cf"),a("d3b7"),a("3ca3"),a("ddb0"),a("2b3d"),a("25f0"),a("ac1f"),a("1276"),a("5319"),{name:"Model",data:function(){return{model_data:[],page_size:10,page_count:1,total_count:1,curr_page:1,task_json_dict:null,model_loader:null,model_list:null,predictHeaders:null,predictData:null,predictResult:null,predictModelId:null,predictModelVisible:!1,is_load:[],timer:null,timer_load_unload:null,predictDataForm:{headers:"",data:"",result:""}}},mounted:function(){null!==window.localStorage.getItem("jeddak_token")&&this.getModelData(this.curr_page)},beforeDestroy:function(){clearInterval(this.timer),clearInterval(this.timer_load_unload)},methods:{getModelData:function(e){var t=this;return Object(n["a"])(regeneratorRuntime.mark((function a(){var r,o,n,l,s,i,d,c;return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:return t.curr_page=e,r=new URLSearchParams({filter:JSON.stringify({deleted:!1}),page_size:t.page_size,curr_page:e,order_by:"-create_time"}),a.prev=2,a.next=5,t.$axios.get("/board/models?"+r.toString(),{headers:{auth:window.localStorage.getItem("jeddak_token")}});case 5:for(o=a.sent,n=o.data,t.model_data=n.data.models,l=0;l<t.model_data.length;l++)t.model_data[l].create_time=new Date(t.model_data[l].create_time+"Z").toLocaleString();t.total_count=n.data.total_count,t.page_count=Math.ceil(t.total_count/t.page_size),a.next=22;break;case 13:if(a.prev=13,a.t0=a["catch"](2),!a.t0.response){a.next=21;break}return console.log(a.t0.response.data),console.log(a.t0.response.status),console.log(a.t0.response.headers),t.$notify.error({title:"提示",message:"操作失败，可能不具备权限，请尝试注销重新登录或联系管理员"}),a.abrupt("return");case 21:console.log(a.t0);case 22:return a.next=24,t.modelList();case 24:s=0,t.is_load=[],i=0;case 27:if(!(i<t.model_data.length)){a.next=43;break}d=!1,c=s;case 30:if(!(c<t.model_list.length)){a.next=38;break}if(t.model_data[i].model_id!==t.model_list[c]){a.next=35;break}return d=!0,s=c+1,a.abrupt("break",38);case 35:c++,a.next=30;break;case 38:t.is_load.push(d),d?t.changeToUnload(i):t.changeToLoad(i);case 40:i++,a.next=27;break;case 43:return a.abrupt("return",t.model_data);case 44:case"end":return a.stop()}}),a,null,[[2,13]])})))()},deleteModel:function(e){var t=this;this.$axios.delete("/board/model/"+e,{headers:{auth:window.localStorage.getItem("jeddak_token")}}).then((function(e){t.$notify.success({title:"提示",message:"模型删除成功，页面刷新中"}),t.getModelData(t.curr_page)})).catch((function(e){console.log(e),t.$notify.error({title:"提示",message:"操作失败，可能不具备权限，请尝试注销重新登录或联系管理员"})}))},changeAlias:function(e){var t=this;this.$axios.patch("/board/model/"+e.model_id,{field:{alias:e.alias}},{headers:{auth:window.localStorage.getItem("jeddak_token")}}).then((function(e){t.$notify.success({title:"提示",message:"修改成功"})})).catch((function(e){console.log(e),t.$notify.error({title:"提示",message:"操作失败，可能不具备权限，请尝试注销重新登录或联系管理员"})}))},handleCurrentChange:function(e){this.getModelData(e)},loadModel:function(e){var t=this;return Object(n["a"])(regeneratorRuntime.mark((function a(){var r,o,n,l,s,i,d,c,u,p,f,h,m;return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:return a.prev=0,a.next=3,t.$axios.get("/board/task/"+e.row.model_id+"?type=detail",{headers:{auth:window.localStorage.getItem("jeddak_token")}});case 3:r=a.sent,o=r.data,t.task_json_dict=JSON.parse(o.data.task_chain),a.next=17;break;case 8:if(a.prev=8,a.t0=a["catch"](0),!a.t0.response){a.next=16;break}return console.log(a.t0.response.data),console.log(a.t0.response.status),console.log(a.t0.response.headers),t.$notify.error({title:"提示",message:"该任务已删除"}),a.abrupt("return");case 16:console.log(a.t0);case 17:if(null===t.task_json_dict||0===t.task_json_dict.length){a.next=35;break}for(n=t.task_json_dict[0].party_names.length,l=[],s=[],i=0;i<n;i++)l.push(e.row.model_id),s.push("load");d=[],c=!1,u=0;case 25:if(!(u<t.task_json_dict.length)){a.next=33;break}if(!t.task_json_dict[u].hasOwnProperty("task_role")){a.next=30;break}for(p=0;p<t.task_json_dict[u].task_role.length;p++)d.push(t.task_json_dict[u].task_role[p]);return c=!0,a.abrupt("break",33);case 30:u++,a.next=25;break;case 33:if(!c)for(d.push("guest"),f=1;f<n;f++)d.push("host");t.model_loader={task_type:"model_loader",task_role:d,model_id:l,action:s};case 35:return t.task_json_dict[0].save_model=!1,a.prev=36,a.next=39,t.$axios.post("/board/predict",{req_type:"model_load",filter:[t.task_json_dict[0],t.model_loader]},{headers:{auth:window.localStorage.getItem("jeddak_token")}});case 39:h=a.sent,m=h.data,console.log(m),200===m.status&&t.$notify.info({title:"提示",message:"模型加载中"}),a.next=54;break;case 45:if(a.prev=45,a.t1=a["catch"](36),!a.t1.response){a.next=53;break}return console.log(a.t1.response.data),console.log(a.t1.response.status),console.log(a.t1.response.headers),t.$notify.error({title:"提示",message:"加载模型失败"}),a.abrupt("return");case 53:console.log(a.t1);case 54:case"end":return a.stop()}}),a,null,[[0,8],[36,45]])})))()},unloadModel:function(e){var t=this;this.$axios.post("/board/predict",{req_type:"model_unload",filter:{model_id:e.row.model_id}},{headers:{auth:window.localStorage.getItem("jeddak_token")}}).then((function(e){})).catch((function(e){console.log(e),t.$notify.error({title:"提示",message:"卸载模型失败"})}))},modelList:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){var a,r;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.prev=0,t.next=3,e.$axios.post("/board/predict",{req_type:"model_list"},{headers:{auth:window.localStorage.getItem("jeddak_token")}});case 3:a=t.sent,r=a.data,e.model_list=r.data,t.next=17;break;case 8:if(t.prev=8,t.t0=t["catch"](0),!t.t0.response){t.next=16;break}return console.log(t.t0.response.data),console.log(t.t0.response.status),console.log(t.t0.response.headers),e.$notify.error({title:"提示",message:"获取加载模型列表失败"}),t.abrupt("return");case 16:console.log(t.t0);case 17:case"end":return t.stop()}}),t,null,[[0,8]])})))()},predictModel:function(e){var t=this;return Object(n["a"])(regeneratorRuntime.mark((function a(){var r,o;return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:return t.predictDataForm.result="正在预测中，请等待",a.prev=1,a.next=4,t.$axios.post("/board/predict",{req_type:"predict",filter:{model_id:e,input_data:[{headers:t.predictHeaders,data:t.predictData},{}]}},{headers:{auth:window.localStorage.getItem("jeddak_token")}});case 4:r=a.sent,o=r.data,t.predictResult=o.data,a.next=18;break;case 9:if(a.prev=9,a.t0=a["catch"](1),!a.t0.response){a.next=17;break}return console.log(a.t0.response.data),console.log(a.t0.response.status),console.log(a.t0.response.headers),t.$notify.error({title:"提示",message:"获取加载模型列表失败"}),a.abrupt("return");case 17:console.log(a.t0);case 18:console.log(t.predictResult["predict_data"]),t.predictDataForm.result=JSON.stringify(t.predictResult["predict_data"]);case 20:case"end":return a.stop()}}),a,null,[[1,9]])})))()},clickLoadAndUnload:function(e){var t=this;this.checkLoadAndUnloadStatus(e.$index)?(this.loadModel(e),this.timer=window.setInterval((function(){return t.modelList()}),1e3),this.timer_load_unload=window.setInterval((function(){return t.isLoadInModelList(e.row.model_id)}),1e3)):(this.unloadModel(e),this.timer=window.setInterval((function(){return t.modelList()}),500),this.timer_load_unload=window.setInterval((function(){return t.isUnloadInModelList(e.row.model_id)}),500))},isLoadInModelList:function(e){var t=this;return Object(n["a"])(regeneratorRuntime.mark((function a(){var r;return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:r=0;case 1:if(!(r<t.model_list.length)){a.next=11;break}if(e!==t.model_list[r]){a.next=8;break}return clearInterval(t.timer),clearInterval(t.timer_load_unload),a.next=7,t.getModelData(t.curr_page);case 7:t.$notify.success({title:"提示",message:"加载模型成功"});case 8:r++,a.next=1;break;case 11:case"end":return a.stop()}}),a)})))()},isUnloadInModelList:function(e){var t=this;return Object(n["a"])(regeneratorRuntime.mark((function a(){var r,o;return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:for(r=!1,o=0;o<t.model_list.length;o++)e===t.model_list[o]&&(r=!0);if(r){a.next=8;break}return clearInterval(t.timer),clearInterval(t.timer_load_unload),a.next=7,t.getModelData(t.curr_page);case 7:t.$notify.success({title:"提示",message:"卸载模型成功"});case 8:case"end":return a.stop()}}),a)})))()},checkLoadAndUnloadStatus:function(e){var t="loadAndUnload"+e,a=this.$refs[t].$el.innerText;return"加载模型"===a},changeToUnload:function(e){var t="loadAndUnload"+e;this.$refs[t].$el.innerText="卸载模型";var a=document.getElementById(t);a.style.backgroundColor="#909399",a.style.borderColor="#909399";var r="predict"+e;a=document.getElementById(r),a.style.backgroundColor="#67C23A",a.style.borderColor="#67C23A"},changeToLoad:function(e){var t="loadAndUnload"+e;this.$refs[t].$el.innerText="加载模型";var a=document.getElementById(t);a.style.backgroundColor="#E6A23C",a.style.borderColor="#E6A23C";var r="predict"+e;a=document.getElementById(r),a.style.backgroundColor="#909399",a.style.borderColor="#909399"},buttonRefAndID:function(e,t){return e+t},clickPredictModel:function(e){this.is_load[e.$index]?(this.predictModelId=e.row.model_id,this.predictModelVisible=!0):this.$notify.info({title:"提示",message:"请先加载模型后再进行预测"})},startPredictButton:function(){this.predictHeaders=this.predictDataForm.headers.replace(/"/g,"").split(",");var e=this.predictDataForm.headers.split("，");this.predictHeaders.length<e.length&&(this.predictHeaders=e);var t=[],a=this.predictDataForm.data.replace(/"/g,"").split(";");"string"===typeof a&&(a=[].push(a));for(var r=0;r<a.length;r++){var o=a[r].split("\n");if("string"===typeof o)t.push(o);else for(var n=0;n<o.length;n++)t.push(o[n])}for(var l=0;l<t.length;l++){var s=t[l].split(","),i=t[l].split("，");s.length<i.length&&(s=i);for(var d=1;d<s.length;d++){var c=parseFloat(s[d]);isNaN(c)||(s[d]=c)}t[l]=s}this.predictData=t,this.predictModel(this.predictModelId)}}}),s=l,i=(a("2b29"),a("2877")),d=Object(i["a"])(s,r,o,!1,null,"77809d61",null);t["default"]=d.exports},ee99:function(e,t,a){}}]);
//# sourceMappingURL=chunk-1056aeb6.97010e66.js.map