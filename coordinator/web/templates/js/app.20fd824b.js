(function(e){function t(t){for(var a,r,i=t[0],l=t[1],u=t[2],c=0,d=[];c<i.length;c++)r=i[c],Object.prototype.hasOwnProperty.call(o,r)&&o[r]&&d.push(o[r][0]),o[r]=0;for(a in l)Object.prototype.hasOwnProperty.call(l,a)&&(e[a]=l[a]);f&&f(t);while(d.length)d.shift()();return s.push.apply(s,u||[]),n()}function n(){for(var e,t=0;t<s.length;t++){for(var n=s[t],a=!0,r=1;r<n.length;r++){var i=n[r];0!==o[i]&&(a=!1)}a&&(s.splice(t--,1),e=l(l.s=n[0]))}return e}var a={},r={app:0},o={app:0},s=[];function i(e){return l.p+"js/"+({about:"about"}[e]||e)+"."+{about:"e63cd2e5","chunk-0ff2cbc4":"0995e48e","chunk-1056aeb6":"97010e66","chunk-273a07dd":"1f289a38","chunk-2f6c3c48":"0d3d5201","chunk-6f0e1069":"b3d995b6","chunk-5608d287":"7e6a778a","chunk-dad67f44":"006e8e35","chunk-784ac172":"90bd4a12"}[e]+".js"}function l(t){if(a[t])return a[t].exports;var n=a[t]={i:t,l:!1,exports:{}};return e[t].call(n.exports,n,n.exports,l),n.l=!0,n.exports}l.e=function(e){var t=[],n={about:1,"chunk-0ff2cbc4":1,"chunk-1056aeb6":1,"chunk-273a07dd":1,"chunk-2f6c3c48":1,"chunk-6f0e1069":1,"chunk-5608d287":1,"chunk-dad67f44":1,"chunk-784ac172":1};r[e]?t.push(r[e]):0!==r[e]&&n[e]&&t.push(r[e]=new Promise((function(t,n){for(var a="css/"+({about:"about"}[e]||e)+"."+{about:"91e1d8ed","chunk-0ff2cbc4":"0e4ab5e9","chunk-1056aeb6":"f2642e65","chunk-273a07dd":"21f8c5df","chunk-2f6c3c48":"a64c21c1","chunk-6f0e1069":"560bf4e8","chunk-5608d287":"e0ad88c2","chunk-dad67f44":"2c6dca5a","chunk-784ac172":"4bc36599"}[e]+".css",o=l.p+a,s=document.getElementsByTagName("link"),i=0;i<s.length;i++){var u=s[i],c=u.getAttribute("data-href")||u.getAttribute("href");if("stylesheet"===u.rel&&(c===a||c===o))return t()}var d=document.getElementsByTagName("style");for(i=0;i<d.length;i++){u=d[i],c=u.getAttribute("data-href");if(c===a||c===o)return t()}var f=document.createElement("link");f.rel="stylesheet",f.type="text/css",f.onload=t,f.onerror=function(t){var a=t&&t.target&&t.target.src||o,s=new Error("Loading CSS chunk "+e+" failed.\n("+a+")");s.code="CSS_CHUNK_LOAD_FAILED",s.request=a,delete r[e],f.parentNode.removeChild(f),n(s)},f.href=o;var h=document.getElementsByTagName("head")[0];h.appendChild(f)})).then((function(){r[e]=0})));var a=o[e];if(0!==a)if(a)t.push(a[2]);else{var s=new Promise((function(t,n){a=o[e]=[t,n]}));t.push(a[2]=s);var u,c=document.createElement("script");c.charset="utf-8",c.timeout=120,l.nc&&c.setAttribute("nonce",l.nc),c.src=i(e);var d=new Error;u=function(t){c.onerror=c.onload=null,clearTimeout(f);var n=o[e];if(0!==n){if(n){var a=t&&("load"===t.type?"missing":t.type),r=t&&t.target&&t.target.src;d.message="Loading chunk "+e+" failed.\n("+a+": "+r+")",d.name="ChunkLoadError",d.type=a,d.request=r,n[1](d)}o[e]=void 0}};var f=setTimeout((function(){u({type:"timeout",target:c})}),12e4);c.onerror=c.onload=u,document.head.appendChild(c)}return Promise.all(t)},l.m=e,l.c=a,l.d=function(e,t,n){l.o(e,t)||Object.defineProperty(e,t,{enumerable:!0,get:n})},l.r=function(e){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},l.t=function(e,t){if(1&t&&(e=l(e)),8&t)return e;if(4&t&&"object"===typeof e&&e&&e.__esModule)return e;var n=Object.create(null);if(l.r(n),Object.defineProperty(n,"default",{enumerable:!0,value:e}),2&t&&"string"!=typeof e)for(var a in e)l.d(n,a,function(t){return e[t]}.bind(null,a));return n},l.n=function(e){var t=e&&e.__esModule?function(){return e["default"]}:function(){return e};return l.d(t,"a",t),t},l.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},l.p="/",l.oe=function(e){throw console.error(e),e};var u=window["webpackJsonp"]=window["webpackJsonp"]||[],c=u.push.bind(u);u.push=t,u=u.slice();for(var d=0;d<u.length;d++)t(u[d]);var f=c;s.push([0,"chunk-vendors"]),n()})({0:function(e,t,n){e.exports=n("56d7")},"034f":function(e,t,n){"use strict";n("85ec")},1001:function(e,t,n){},"56d7":function(e,t,n){"use strict";n.r(t);n("e260"),n("e6cf"),n("cca6"),n("a79d"),n("4de4");var a=n("2b0e"),r=function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("div",{attrs:{id:"app"}},[n("el-menu",{staticClass:"el-menu-demo",attrs:{"default-active":e.activeIndex,mode:"horizontal",id:"header-menu"},on:{select:e.handleSelect}},[n("el-menu-item",{attrs:{index:"/"}},[e._v("JEDDAK")]),n("el-menu-item",{attrs:{index:"/task"}},[e._v("任务管理")]),n("el-menu-item",{attrs:{index:"/model"}},[e._v("模型管理")]),n("el-menu-item",{attrs:{index:"/data"}},[e._v("数据管理")]),this.$global.isAdmin?n("el-menu-item",{attrs:{index:"/admin"}},[e._v("用户管理")]):e._e(),n("el-menu-item",{attrs:{index:"/about"}},[e._v("使用指南")])],1),n("router-view")],1)},o=[],s={name:"App",data:function(){return{activeIndex:null}},created:function(){var e=this;this.checkLocalToken(),window.localStorage.getItem("jeddak_isAdmin")&&(this.$global.isAdmin=JSON.parse(window.localStorage.getItem("jeddak_isAdmin"))),window.localStorage.getItem("jeddak_partyNumber")&&(this.$global.partyNumber=JSON.parse(window.localStorage.getItem("jeddak_partyNumber"))),window.localStorage.getItem("jeddak_taskChain")&&(this.$global.taskChainForm=JSON.parse(window.localStorage.getItem("jeddak_taskChain"))),window.addEventListener("beforeunload",(function(){window.localStorage.setItem("jeddak_isAdmin",JSON.stringify(e.$global.isAdmin)),window.localStorage.setItem("jeddak_partyNumber",JSON.stringify(e.$global.partyNumber)),window.localStorage.setItem("jeddak_taskChain",JSON.stringify(e.$global.taskChainForm))}))},updated:function(){this.checkLocalToken()},methods:{checkLocalToken:function(){this.activeIndex=this.$route.path;var e=window.localStorage.getItem("jeddak_token");"/login"===this.$route.path?null!==e&&this.$router.push("/"):null===e&&this.$router.push("/login")},handleSelect:function(e,t){this.$route.path!==t[0]&&this.$router.push(t[0])}}},i=s,l=(n("034f"),n("2877")),u=Object(l["a"])(i,r,o,!1,null,null,null),c=u.exports,d=(n("d3b7"),n("3ca3"),n("ddb0"),n("8c4f")),f=function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("div",{staticClass:"home"},[n("el-row",{attrs:{id:"top-buttons"}},[n("el-button",{attrs:{type:"primary"},on:{click:function(t){e.changePasswordDialogVisible=!0}}},[e._v(" 修改密码 "),n("i",{staticClass:"el-icon-key"})]),n("el-button",{on:{click:e.logout}},[e._v(" 注销登录 "),n("i",{staticClass:"el-icon-switch-button"})])],1),n("el-row",{attrs:{gutter:20}},[n("el-col",{attrs:{span:24,xs:24,sm:12}},[n("el-card",{staticClass:"box-card"},[n("div",{staticClass:"clearfix",attrs:{slot:"header"},slot:"header"},[n("b",[e._v("账号信息")])]),n("div",[e._v("参与方名称："+e._s(e.party_info.party_name||"未声明"))]),n("hr"),n("div",[e._v("通信组件类型："+e._s(e.party_info.syncer_type||"未声明"))]),n("hr"),n("div",[e._v("通信组件状态："+e._s(e.party_info.syncer_status||"未连接"))])])],1),n("el-col",{attrs:{span:24,xs:24,sm:12}},[n("el-card",{staticClass:"box-card"},[n("div",{staticClass:"clearfix",attrs:{slot:"header"},slot:"header"},[n("b",[e._v("任务信息")])]),n("div",[e._v("成功任务数："+e._s(e.task_info.finish||0))]),n("hr"),n("div",[e._v("失败任务数："+e._s(e.task_info.error||0))]),n("hr"),n("div",[e._v("未完成任务数："+e._s(e.task_info.ready||0+e.task_info.run||0))])])],1)],1),n("el-dialog",{attrs:{title:"修改密码",visible:e.changePasswordDialogVisible,"before-close":e.changePasswordDialogClose,width:"30%"}},[n("el-form",{ref:"form",attrs:{model:e.form,rules:e.rules}},[n("el-form-item",{attrs:{prop:"password"}},[n("el-input",{attrs:{placeholder:"请输入原密码","show-password":""},nativeOn:{keyup:function(t){return!t.type.indexOf("key")&&e._k(t.keyCode,"enter",13,t.key,"Enter")?null:e.onSubmit.apply(null,arguments)}},model:{value:e.form.password,callback:function(t){e.$set(e.form,"password",t)},expression:"form.password"}})],1),n("el-form-item",{attrs:{prop:"password"}},[n("el-input",{attrs:{placeholder:"请输入新密码","show-password":""},nativeOn:{keyup:function(t){return!t.type.indexOf("key")&&e._k(t.keyCode,"enter",13,t.key,"Enter")?null:e.onSubmit.apply(null,arguments)}},model:{value:e.form.new_password,callback:function(t){e.$set(e.form,"new_password",t)},expression:"form.new_password"}})],1)],1),n("span",{staticClass:"dialog-footer",attrs:{slot:"footer"},slot:"footer"},[n("el-button",{attrs:{type:"primary"},on:{click:e.changePassword}},[e._v("修改")]),n("el-button",{attrs:{type:"info"},on:{click:e.changePasswordDialogClose}},[e._v("取消")])],1)],1)],1)},h=[],p=n("1da1"),m=(n("96cf"),n("2b3d"),n("25f0"),n("6c27")),g={name:"Home",data:function(){return{form:{username:"",password:"",new_password:""},password:"",new_password:"",salt:"",new_salt:"",rules:{username:[{required:!0,trigger:"blur"}],password:[{required:!0,trigger:"blur"}],new_password:[{required:!0,trigger:"blur"}]},changePasswordDialogVisible:!1,party_info:{party_name:"",syncer_type:"",syncer_status:""},task_info:{task_num:0,finished_num:0,unfinished_num:0},party_init_form:{party_name:"",syncer_server:"",syncer_type:""},clear_topic_form:{syncer_server:"",syncer_type:""},loading:!1}},mounted:function(){null!==window.localStorage.getItem("jeddak_token")&&(this.fetch_party_info(),this.fetch_task_info()),null!==window.localStorage.getItem("jeddak_username")&&(this.form.username=window.localStorage.getItem("jeddak_username"))},methods:{fetch_party_info:function(){var e=this;this.$axios.get("/board/party",{headers:{auth:window.localStorage.getItem("jeddak_token")}}).then((function(t){e.party_info.party_name=t.data.data.party_name,e.party_info.syncer_type=t.data.data.syncer_type,""!==t.data.data.party_name&&(e.party_info.syncer_status="已连接")})).catch((function(t){console.log(t),e.$notify.error({title:"提示",message:"操作失败，可能不具备权限，请尝试注销重新登录或联系管理员"})}))},fetch_task_info:function(){var e=this,t=new URLSearchParams({filter:JSON.stringify({deleted:!1}),req_type:"stat"});this.$axios.get("/board/tasks?"+t.toString(),{headers:{auth:window.localStorage.getItem("jeddak_token")}}).then((function(t){void 0!==t.data.data&&(e.task_info=t.data.data)})).catch((function(t){console.log(t),e.$notify.error({title:"提示",message:"操作失败，可能不具备权限，请尝试注销重新登录或联系管理员"})}))},logout:function(){window.localStorage.removeItem("jeddak_token"),window.localStorage.removeItem("jeddak_isAdmin"),this.$global.isAdmin=!1,this.$router.push("/login")},changePassword:function(){var e=this;return Object(p["a"])(regeneratorRuntime.mark((function t(){var n,a;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.prev=0,t.next=3,e.$axios.post("/board/user",{filter:{name:e.form.username},req_type:"get_change_password_salt"});case 3:n=t.sent,a=n.data,""!==a.data.salt&&(e.salt=a.data.salt,e.new_salt=a.data.new_salt),t.next=12;break;case 8:t.prev=8,t.t0=t["catch"](0),t.t0.response&&(console.log(t.t0.response.data),console.log(t.t0.response.status),console.log(t.t0.response.headers),e.$notify.error({title:"提示",message:"获取盐值失败"}),e.loading=!1),console.log(t.t0);case 12:e.password=Object(m["sha256"])(e.salt+e.form.password).toString(),e.new_password=Object(m["sha256"])(e.new_salt+e.form.new_password).toString(),e.$axios.patch("/board/user",{req_type:"update_password",filter:{name:e.form.username,password:e.password},field:{password:e.new_password,salt:e.new_salt}},{headers:{auth:window.localStorage.getItem("jeddak_token")}}).then((function(t){console.log(t),e.loading=!1,window.localStorage.setItem("jeddak_token",t.data.data.auth),e.$notify.success({title:"提示",message:"修改成功"})})).catch((function(t){console.log(t),e.$notify.error({title:"提示",message:"修改失败"}),e.loading=!1})),e.changePasswordDialogVisible=!1,e.form.password="",e.form.new_password="";case 18:case"end":return t.stop()}}),t,null,[[0,8]])})))()},changePasswordDialogClose:function(){this.form.password="",this.form.new_password="",this.changePasswordDialogVisible=!1}}},w=g,k=(n("c607"),Object(l["a"])(w,f,h,!1,null,"a3db45c6",null)),_=k.exports;a["default"].use(d["a"]);var b,v,y=[{path:"/",name:"Home",component:_},{path:"/about",name:"About",component:function(){return n.e("about").then(n.bind(null,"f820"))}},{path:"/login",name:"Login",component:function(){return n.e("chunk-273a07dd").then(n.bind(null,"a55b"))}},{path:"/task",name:"Task",component:function(){return n.e("chunk-2f6c3c48").then(n.bind(null,"1d21"))}},{path:"/task_new",name:"TaskNew",component:function(){return Promise.all([n.e("chunk-6f0e1069"),n.e("chunk-dad67f44")]).then(n.bind(null,"8f5e"))}},{path:"/task_detail",name:"TaskDetail",component:function(){return Promise.all([n.e("chunk-6f0e1069"),n.e("chunk-5608d287")]).then(n.bind(null,"8b46"))}},{path:"/model",name:"Model",component:function(){return n.e("chunk-1056aeb6").then(n.bind(null,"b679"))}},{path:"/data",name:"Data",component:function(){return n.e("chunk-784ac172").then(n.bind(null,"9352"))}},{path:"/admin",name:"Admin",component:function(){return n.e("chunk-0ff2cbc4").then(n.bind(null,"3530"))}}],S=new d["a"]({base:"/",routes:y}),I=S,x=n("5c96"),j=n.n(x),O=(n("0fae"),n("bc3a")),$=n.n(O),P=n("4eb5"),C=n.n(P),N=(n("ac1f"),n("1276"),!1),A={name:"Global",isAdmin:N,partyNumber:null,moduleTable:[],indexOfModuleTable:0,isEdit:!1,taskChainForm:[],moduleDialogButtonLabel:function(e){return e?"修改":"添加"},checkPositiveInteger:function(e){return null!==e&&/^[1-9]\d*$/.test(e)},checkE02E1:function(e){if(this.checkIsValue(e)){var t=parseFloat(e);if(t>=0&&t<=1)return!0}return!1},checkL02E1:function(e){if(this.checkIsValue(e)){var t=parseFloat(e);if(t>0&&t<=1)return!0}return!1},checkL02S1:function(e){if(this.checkIsValue(e)){var t=parseFloat(e);if(t>0&&t<1)return!0}return!1},checkGE0:function(e){if(this.checkIsValue(e)){var t=parseFloat(e);if(t>=0)return!0}return!1},checkGE2:function(e){if(this.checkIsValue(e)){var t=parseFloat(e);if(t>=2)return!0}return!1},checkGL0:function(e){if(this.checkIsValue(e)){var t=parseFloat(e);if(t>0)return!0}return!1},checkIsValue:function(e){return null!==e&&/^\d+(\.\d+)?$/.test(e)},formatPartyPara:function(e,t,n,a,r,o){var s=[];e=[],t=[],n=[];for(var i=0;i<a;i++){e.push(null);var l={name:o+i};t.push(l);var u="第"+(i+1)+"个参与方"+r;n.push(u)}return s.push(e),s.push(t),s.push(n),s},formatPartyParaControlShow:function(e,t,n,a,r,o){var s=[];e=[],t=[],n=[];for(var i=0;i<a;i++){e.push(null);var l={name:o+i,isShow:!1,isFirstTime:!0};t.push(l);var u="第"+(i+1)+"个参与方"+r;n.push(u)}return s.push(e),s.push(t),s.push(n),s},formatList:function(e,t){e=[];for(var n=0;n<t;n++)e.push(null);return e},copyJson:function(e){return JSON.parse(JSON.stringify(e))},writeLocalStorage:function(e,t){window.localStorage.setItem(e,JSON.stringify(t))},readLocalStorage:function(e){return window.localStorage.getItem(e)?JSON.parse(window.localStorage.getItem(e)):null},parseInput2IntArray:function(e){if(null===e)return null;e+="";var t=e.split(","),n=e.split("，");t.length<n.length&&(t=n);for(var a=0;a<t.length;a++)t[a]=parseInt(t[a]);return t},parseStrArray2IntArray:function(e){if(null===e)return e;if("string"==typeof e){var t=parseInt(e),n=parseFloat(e),a=t;return t!==n&&(a=n),isNaN(a)||(e=a),e}for(var r=0;r<e.length;r++)if(null!==e[r]){var o=parseInt(e[r]),s=parseFloat(e[r]),i=o;o!==s&&(i=s),isNaN(i)||(e[r]=i)}return e},array2str:function(e){if(null===e)return null;for(var t=e[0],n=1;n<e.length;n++)t=t+","+e[n];return t},array2strWithSeg:function(e,t){if(null===e)return null;for(var n=e[0]+"",a=1;a<e.length;a++)n=n+t+e[a];return n},handleIsShowPara:function(e,t,n){for(var a=[],r=0;r<e.length;r++)if(t[r].isShow){if(null===e[r]||""===e[r]){a.push(null);continue}"int"===n?a.push(parseInt(e[r])):"float"===n?a.push(parseFloat(e[r])):a.push(e[r])}else a.push(null);return a}},E=A,L=Object(l["a"])(E,b,v,!1,null,null,null),D=L.exports,T=n("5c7f");a["default"].config.productionTip=!1,a["default"].use(j.a),a["default"].prototype.$axios=$.a,a["default"].use(C.a),a["default"].prototype.$global=D,a["default"].component("v-chart",T["a"]),a["default"].filter("task_status",(function(e){switch(e){case"finish":return"成功";case"error":return"失败";case"ready":return"就绪";case"run":return"进行中";default:return"未知"}})),new a["default"]({router:I,render:function(e){return e(c)}}).$mount("#app"),document.title="JEDDAK协同计算平台"},"85ec":function(e,t,n){},c607:function(e,t,n){"use strict";n("1001")}});
//# sourceMappingURL=app.20fd824b.js.map