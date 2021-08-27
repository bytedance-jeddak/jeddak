(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-0ff2cbc4"],{3530:function(e,t,o){"use strict";o.r(t);var r=function(){var e=this,t=e.$createElement,o=e._self._c||t;return o("div",{staticClass:"admin"},[o("el-row",{attrs:{id:"top-buttons"}},[o("el-button",{staticClass:"oper-btn",attrs:{type:"primary"},on:{click:e.addUser}},[e._v(" 添加用户 "),o("i",{staticClass:"el-icon-circle-plus"})]),o("el-button",{staticClass:"oper-btn",attrs:{type:"danger"},on:{click:function(t){e.deleteUserDialogVisible=!0}}},[e._v(" 删除用户 "),o("i",{staticClass:"el-icon-remove"})])],1),o("el-card",{staticClass:"box-card"},[o("el-table",{staticStyle:{width:"100%"},attrs:{data:e.userTable,stripe:""}},[o("el-table-column",{attrs:{type:"index"}}),o("el-table-column",{attrs:{prop:"name",label:"用户名"}}),o("el-table-column",{attrs:{label:"角色"},scopedSlots:e._u([{key:"default",fn:function(t){return[e._v(" "+e._s(e.getRolesDescription(t.row["role"]))+" ")]}}])}),o("el-table-column",{attrs:{label:"操作"},scopedSlots:e._u([{key:"default",fn:function(t){return[o("el-tooltip",{attrs:{content:"修改",placement:"top",enterable:!1}},[o("el-button",{attrs:{type:"primary",icon:"el-icon-edit",size:"mini"},on:{click:function(o){return e.modifyUser(t)}}})],1),o("el-tooltip",{attrs:{content:"删除",placement:"top",enterable:!1}},[o("el-button",{attrs:{type:"danger",icon:"el-icon-delete",size:"mini"},on:{click:function(o){return e.deleteUser(t.row.name)}}})],1)]}}])})],1),o("el-pagination",{attrs:{id:"pagination",background:"","page-size":e.page_size,"page-count":e.page_count,total:e.total_count,layout:"prev, pager, next, jumper"},on:{"current-change":e.handleCurrentChange}})],1),o("el-dialog",{attrs:{title:"添加用户",visible:e.addUserDialogVisible,"before-close":e.cancelAddUser,width:"35%"}},[o("el-form",{ref:"form",attrs:{"label-position":e.formPosition,model:e.addUserForm,"label-width":"80px"}},[o("el-form-item",{attrs:{label:"用户名"}},[o("el-input",{model:{value:e.addUserForm.username,callback:function(t){e.$set(e.addUserForm,"username",t)},expression:"addUserForm.username"}})],1),o("el-form-item",{attrs:{label:"密码"}},[o("el-input",{model:{value:e.addUserForm.password,callback:function(t){e.$set(e.addUserForm,"password",t)},expression:"addUserForm.password"}})],1),o("el-form-item",{attrs:{label:"角色"}},[o("el-select",{attrs:{placeholder:"请选择角色"},model:{value:e.addUserForm.role,callback:function(t){e.$set(e.addUserForm,"role",t)},expression:"addUserForm.role"}},e._l(e.getRoles(),(function(e){return o("el-option",{key:e.value,attrs:{label:e.label,value:e.value}})})),1)],1),o("el-form-item",[o("el-button",{attrs:{type:"primary"},on:{click:e.addUserFormButton}},[e._v("增加")]),o("el-button",{attrs:{type:"info"},on:{click:e.cancelAddUserFormButton}},[e._v("取消")])],1)],1)],1),o("el-dialog",{attrs:{title:"修改用户",visible:e.modifyUserDialogVisible,"before-close":e.cancelModifyUser,width:"30%"}},[o("el-form",{ref:"form",attrs:{"label-position":e.formPosition,model:e.modifyUserForm,"label-width":"80px"}},[o("el-form-item",{attrs:{label:"用户名"}},[o("el-input",{attrs:{disabled:""},model:{value:this.modifyUsername,callback:function(t){e.$set(this,"modifyUsername",t)},expression:"this.modifyUsername"}})],1),o("el-form-item",{attrs:{label:"密码"}},[o("el-input",{model:{value:e.modifyUserForm.password,callback:function(t){e.$set(e.modifyUserForm,"password",t)},expression:"modifyUserForm.password"}})],1),o("el-form-item",{attrs:{label:"角色"}},[o("el-select",{attrs:{placeholder:"请选择角色"},model:{value:e.modifyUserForm.role,callback:function(t){e.$set(e.modifyUserForm,"role",t)},expression:"modifyUserForm.role"}},e._l(e.getRoles(),(function(e){return o("el-option",{key:e.value,attrs:{label:e.label,value:e.value}})})),1)],1),o("el-form-item",[o("el-button",{attrs:{type:"primary"},on:{click:e.modifyUserFormButton}},[e._v("修改")]),o("el-button",{attrs:{type:"info"},on:{click:e.cancelModifyUserFormButton}},[e._v("取消")])],1)],1)],1),o("el-dialog",{attrs:{title:"删除用户",visible:e.deleteUserDialogVisible,"before-close":e.cancelDeleteUser,width:"30%"}},[o("el-form",{ref:"form",attrs:{"label-position":e.formPosition,model:e.deleteUserForm,"label-width":"80px"}},[o("el-form-item",{attrs:{label:"用户名："}},[o("el-input",{model:{value:e.deleteUserForm.name,callback:function(t){e.$set(e.deleteUserForm,"name",t)},expression:"deleteUserForm.name"}})],1),o("el-form-item",[o("el-button",{attrs:{type:"primary"},on:{click:function(t){return e.deleteUser(e.deleteUserForm.name)}}},[e._v("删除")]),o("el-button",{attrs:{type:"info"},on:{click:e.cancelDeleteUserFormButton}},[e._v("取消")])],1)],1)],1)],1)},s=[],a=o("1da1"),l=(o("96cf"),o("d3b7"),o("25f0"),o("b0c0"),o("6c27")),n={name:"Admin",data:function(){return{userTable:[],addUserDialogVisible:!1,addUserForm:{username:"",password:"",role:null},formPosition:"left",modifyUserForm:{password:"",role:null},modifyUsername:"",modifyUserDialogVisible:!1,deleteUserDialogVisible:!1,deleteUserForm:{name:""},adminRole:32,salt:"",new_salt:"",roles1:[{value:2,label:"管理员"},{value:4,label:"一级用户"},{value:8,label:"二级用户"},{value:16,label:"三级用户"},{value:32,label:"四级用户"},{value:64,label:"五级用户"}],roles2:[{value:4,label:"一级用户"},{value:8,label:"二级用户"},{value:16,label:"三级用户"},{value:32,label:"四级用户"},{value:64,label:"五级用户"}],page_size:10,page_count:1,total_count:1,current_page:1}},created:function(){this.checkAdmin()},mounted:function(){this.getUserList(1)},methods:{checkAdmin:function(){var e=this;this.$axios.post("/board/user/admin_auth",{},{headers:{auth:window.localStorage.getItem("jeddak_token")}}).then((function(t){e.adminRole=t.data.data.role})).catch((function(t){console.log(t),e.$router.push("/")}))},addUser:function(){this.addUserDialogVisible=!0},deleteUser:function(e){var t=this;this.$axios.post("/board/user",{filter:{name:e},req_type:"delete_user"},{headers:{auth:window.localStorage.getItem("jeddak_token")}}).then((function(e){console.log(e),t.$notify.success({title:"提示",message:"删除成功"}),t.loading=!1})).catch((function(e){console.log(e),t.$notify.error({title:"提示",message:"获取用户列表失败"}),t.loading=!1})),this.getUserList(this.current_page),this.deleteUserDialogVisible=!1},cancelAddUser:function(){this.addUserDialogVisible=!1},cancelModifyUser:function(){this.modifyUserDialogVisible=!1},cancelDeleteUser:function(){this.deleteUserDialogVisible=!1},getRoles:function(){return 1===this.adminRole?this.roles1:2===this.adminRole?this.roles2:void 0},getRolesDescription:function(e){switch(e){case 1:return"一级管理员";case 2:return"二级管理员";case 4:return"一级用户";case 8:return"二级用户";case 16:return"三级用户";case 32:return"四级用户";case 64:return"五级用户"}},getUserList:function(e){var t=this;this.$axios.post("/board/user",{req_type:"get_user_list",filter:JSON.stringify({deleted:!1}),page_size:this.page_size+1,curr_page:e},{headers:{auth:window.localStorage.getItem("jeddak_token")}}).then((function(e){console.log(e.data),t.userTable=e.data.data.users,t.total_count=e.data.data.total_count,t.page_count=Math.ceil(t.total_count/t.page_size),t.loading=!1})).catch((function(e){console.log(e),t.$notify.error({title:"提示",message:"获取用户列表失败"}),t.loading=!1}))},addUserFormButton:function(){var e=this;return Object(a["a"])(regeneratorRuntime.mark((function t(){var o,r,s,a;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return e.loading=!0,t.prev=1,t.next=4,e.$axios.post("/board/user",{filter:{name:e.addUserForm.username,role:e.addUserForm.role},req_type:"get_add_user_salt"},{headers:{auth:window.localStorage.getItem("jeddak_token")}});case 4:o=t.sent,r=o.data,""!==r.data.salt&&(console.log(r.data.salt),e.salt=r.data.salt),t.next=13;break;case 9:t.prev=9,t.t0=t["catch"](1),t.t0.response&&(console.log(t.t0.response.data),console.log(t.t0.response.status),console.log(t.t0.response.headers)),console.log(t.t0);case 13:return t.prev=13,t.next=16,e.$axios.post("/board/user",{filter:{name:e.addUserForm.username,password:Object(l["sha256"])(e.salt+e.addUserForm.password).toString(),salt:e.salt,role:e.addUserForm.role},req_type:"add_user"},{headers:{auth:window.localStorage.getItem("jeddak_token")}});case 16:s=t.sent,a=s.data,200===a.status&&(e.loading=!1,e.$notify.success({title:"提示",message:"添加成功"}),e.loading=!1),t.next=25;break;case 21:t.prev=21,t.t1=t["catch"](13),t.t1.response&&(console.log(t.t1.response.data),console.log(t.t1.response.status),console.log(t.t1.response.headers),e.$notify.error({title:"提示",message:"添加用户失败，用户名已存在"}),e.loading=!1),console.log(t.t1);case 25:e.cleanAddUserForm(),e.getUserList(e.current_page),e.addUserDialogVisible=!1;case 28:case"end":return t.stop()}}),t,null,[[1,9],[13,21]])})))()},cancelAddUserFormButton:function(){this.cleanAddUserForm(),this.addUserDialogVisible=!1},cancelModifyUserFormButton:function(){this.cleanModifyUserForm(),this.modifyUserDialogVisible=!1},cancelDeleteUserFormButton:function(){this.cleanDeleteUserForm(),this.deleteUserDialogVisible=!1},cleanDeleteUserForm:function(){this.deleteUserForm={name:""}},handleCurrentChange:function(e){this.getUserList(e),this.current_page=e},modifyUser:function(e){this.modifyUserDialogVisible=!0,this.modifyUsername=e.row.name},modifyUserFormButton:function(){var e=this;return Object(a["a"])(regeneratorRuntime.mark((function t(){var o,r;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.prev=0,t.next=3,e.$axios.post("/board/user",{filter:{name:e.modifyUsername},req_type:"get_change_password_salt"});case 3:o=t.sent,r=o.data,""!==r.data.salt&&(e.new_salt=r.data.new_salt),t.next=12;break;case 8:t.prev=8,t.t0=t["catch"](0),t.t0.response&&(console.log(t.t0.response.data),console.log(t.t0.response.status),console.log(t.t0.response.headers),e.$notify.error({title:"提示",message:"获取盐值失败"}),e.loading=!1),console.log(t.t0);case 12:e.new_password=Object(l["sha256"])(e.new_salt+e.modifyUserForm.password).toString(),""===e.modifyUserForm.password&&(e.new_password=null,e.new_salt=null),e.$axios.patch("/board/user",{req_type:"update_user",filter:{name:e.modifyUsername},field:{password:e.new_password,salt:e.new_salt,role:e.modifyUserForm.role}},{headers:{auth:window.localStorage.getItem("jeddak_token")}}).then((function(t){console.log(t),e.loading=!1,e.$notify.success({title:"提示",message:"修改成功"})})).catch((function(t){console.log(t),e.$notify.error({title:"提示",message:"修改失败"}),e.loading=!1})),e.getUserList(e.current_page),e.cleanModifyUserForm(),e.modifyUserDialogVisible=!1;case 18:case"end":return t.stop()}}),t,null,[[0,8]])})))()},cleanAddUserForm:function(){this.addUserForm={username:"",password:"",role:null}},cleanModifyUserForm:function(){this.modifyUserForm={password:"",role:null}}}},i=n,c=(o("573e"),o("2877")),d=Object(c["a"])(i,r,s,!1,null,"8744d336",null);t["default"]=d.exports},"573e":function(e,t,o){"use strict";o("be50")},b0c0:function(e,t,o){var r=o("83ab"),s=o("9bf2").f,a=Function.prototype,l=a.toString,n=/^\s*function ([^ (]*)/,i="name";r&&!(i in a)&&s(a,i,{configurable:!0,get:function(){try{return l.call(this).match(n)[1]}catch(e){return""}}})},be50:function(e,t,o){}}]);
//# sourceMappingURL=chunk-0ff2cbc4.0995e48e.js.map